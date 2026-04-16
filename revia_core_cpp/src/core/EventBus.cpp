// Phase 1 - EventBus implementation.
#include "core/EventBus.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

// ---------------------------------------------------------------------------
// MpmcBoundedQueue
// ---------------------------------------------------------------------------

template <typename T>
EventBus::MpmcBoundedQueue<T>::MpmcBoundedQueue(std::size_t capacity)
    : capacity_(round_up_power_of_two(capacity == 0 ? 1 : capacity)),
      mask_(capacity_ - 1),
      buffer_(std::make_unique<Cell[]>(capacity_)) {
    for (std::size_t i = 0; i < capacity_; ++i) {
        buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }
}

template <typename T>
std::size_t EventBus::MpmcBoundedQueue<T>::round_up_power_of_two(std::size_t value) {
    std::size_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

template <typename T>
bool EventBus::MpmcBoundedQueue<T>::enqueue(T item) {
    Cell* cell = nullptr;
    std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

    for (;;) {
        cell = &buffer_[pos & mask_];
        const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
        const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);

        if (diff == 0) {
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
        } else if (diff < 0) {
            return false;
        } else {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }

    cell->data = std::move(item);
    cell->sequence.store(pos + 1, std::memory_order_release);
    return true;
}

template <typename T>
bool EventBus::MpmcBoundedQueue<T>::dequeue(T& item) {
    Cell* cell = nullptr;
    std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

    for (;;) {
        cell = &buffer_[pos & mask_];
        const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
        const auto diff = static_cast<std::intptr_t>(seq) -
                          static_cast<std::intptr_t>(pos + 1);

        if (diff == 0) {
            if (dequeue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
        } else if (diff < 0) {
            return false;
        } else {
            pos = dequeue_pos_.load(std::memory_order_relaxed);
        }
    }

    item = std::move(cell->data);
    cell->sequence.store(pos + capacity_, std::memory_order_release);
    return true;
}

template <typename T>
bool EventBus::MpmcBoundedQueue<T>::empty() const {
    return enqueue_pos_.load(std::memory_order_acquire) ==
           dequeue_pos_.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

EventBus::EventBus(std::size_t queue_capacity)
    : queue_(queue_capacity) {}

EventBus::~EventBus() {
    stop();
}

void EventBus::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }

    worker_ = std::thread([this] { worker_loop(); });
    StructuredLogger::instance().info("event_bus.started", {});
}

void EventBus::stop() {
    const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
    if (!was_running && !worker_.joinable()) {
        return;
    }

    wake_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }

    StructuredLogger::instance().info("event_bus.stopped", {
        {"published", published_count_.load(std::memory_order_relaxed)},
        {"dispatched", dispatched_count_.load(std::memory_order_relaxed)},
        {"dropped", dropped_count_.load(std::memory_order_relaxed)}
    });
}

bool EventBus::publish(IEvent event) {
    normalize(event);

    if (!queue_.enqueue(std::move(event))) {
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
        StructuredLogger::instance().warn("event_bus.queue_full", {});
        return false;
    }

    published_count_.fetch_add(1, std::memory_order_relaxed);
    wake_cv_.notify_one();
    return true;
}

EventBus::SubscriptionId EventBus::subscribe(EventType type, EventHandler handler) {
    if (!handler) {
        throw std::invalid_argument("EventBus::subscribe requires a handler");
    }

    const auto id = next_subscription_id_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(subscribers_mtx_);
        subscribers_[type].push_back(Subscriber{id, std::move(handler)});
    }

    StructuredLogger::instance().info("event_bus.subscribed", {
        {"event_type", std::string(to_string(type))},
        {"subscription_id", id}
    });
    return id;
}

bool EventBus::unsubscribe(EventType type, SubscriptionId id) {
    std::lock_guard<std::mutex> lock(subscribers_mtx_);
    auto it = subscribers_.find(type);
    if (it == subscribers_.end()) {
        return false;
    }

    auto& handlers = it->second;
    const auto old_size = handlers.size();
    handlers.erase(
        std::remove_if(
            handlers.begin(),
            handlers.end(),
            [id](const Subscriber& sub) { return sub.id == id; }),
        handlers.end());

    const bool removed = handlers.size() != old_size;
    if (handlers.empty()) {
        subscribers_.erase(it);
    }

    if (removed) {
        StructuredLogger::instance().info("event_bus.unsubscribed", {
            {"event_type", std::string(to_string(type))},
            {"subscription_id", id}
        });
    }
    return removed;
}

void EventBus::worker_loop() {
    while (running_.load(std::memory_order_acquire) || !queue_.empty()) {
        IEvent event;
        if (!queue_.dequeue(event)) {
            std::unique_lock<std::mutex> lock(wake_mtx_);
            wake_cv_.wait_for(lock, std::chrono::milliseconds(25), [this] {
                return !running_.load(std::memory_order_acquire) || !queue_.empty();
            });
            continue;
        }

        StructuredLogger::instance().debug("event_bus.event_dequeued", {
            {"event_id", event.id},
            {"event_type", std::string(to_string(event.type))},
            {"source", std::string(to_string(event.source))},
            {"correlation_id", event.correlation_id}
        });
        dispatch(event);
        dispatched_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

void EventBus::dispatch(const IEvent& event) {
    std::vector<Subscriber> handlers;
    {
        std::lock_guard<std::mutex> lock(subscribers_mtx_);
        auto it = subscribers_.find(event.type);
        if (it != subscribers_.end()) {
            handlers = it->second;
        }
    }

    if (handlers.empty()) {
        StructuredLogger::instance().debug("event_bus.no_subscribers", {
            {"event_id", event.id},
            {"event_type", std::string(to_string(event.type))}
        });
        return;
    }

    for (const auto& sub : handlers) {
        try {
            sub.handler(event);
        } catch (const std::exception& exc) {
            StructuredLogger::instance().error("event_bus.handler_error", {
                {"event_id", event.id},
                {"event_type", std::string(to_string(event.type))},
                {"subscription_id", sub.id},
                {"error", exc.what()}
            });
        } catch (...) {
            StructuredLogger::instance().error("event_bus.handler_error", {
                {"event_id", event.id},
                {"event_type", std::string(to_string(event.type))},
                {"subscription_id", sub.id},
                {"error", "unknown exception"}
            });
        }
    }
}

void EventBus::normalize(IEvent& event) const {
    if (event.id.empty()) {
        event.id = make_event_id();
    }
    if (event.correlation_id.empty()) {
        event.correlation_id = event.id;
    }
    if (event.created_at == Timestamp{}) {
        event.created_at = now();
    }
}

std::string EventBus::make_event_id() {
    static std::atomic<std::uint64_t> counter{1};
    thread_local std::mt19937_64 rng{std::random_device{}()};

    const auto c = counter.fetch_add(1, std::memory_order_relaxed);
    const auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto random = rng();

    std::ostringstream oss;
    oss << "evt-" << std::hex << ticks << '-' << random << '-' << c;
    return oss.str();
}

} // namespace revia::core

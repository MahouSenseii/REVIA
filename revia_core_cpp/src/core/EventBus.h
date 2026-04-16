// Phase 1 - EventBus.
//
// Single ownership:
//   * EventBus owns event normalization, queueing, and subscriber dispatch.
//   * It does not decide behavior, mutate runtime state, or execute actions.
//
// Design notes:
//   * publish(...) is non-blocking with respect to event delivery. It normalizes
//     the value event, attempts to enqueue, and returns false if the queue is full.
//   * A single worker thread drains the queue and invokes subscribers. This keeps
//     producers isolated from handler latency during the strangler-fig migration.
//   * Subscriber table mutation is mutex-protected; event delivery copies the
//     current handlers before invoking them so handlers can subscribe/unsubscribe
//     safely from another thread.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "interfaces/IEvent.h"

namespace revia::core {

class EventBus {
public:
    using EventHandler = std::function<void(const IEvent&)>;
    using SubscriptionId = std::uint64_t;

    explicit EventBus(std::size_t queue_capacity = 1024);
    ~EventBus();

    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;

    // Start the dispatch worker. Safe to call more than once.
    void start();

    // Stop the dispatch worker and drain already-queued events before joining.
    void stop();

    [[nodiscard]] bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

    // Normalize and enqueue an event. Returns false if the queue is full.
    [[nodiscard]] bool publish(IEvent event);

    // Subscribe to one event type. The returned id can be used to unsubscribe.
    [[nodiscard]] SubscriptionId subscribe(EventType type, EventHandler handler);

    // Remove a subscription. Returns true when a matching subscription existed.
    [[nodiscard]] bool unsubscribe(EventType type, SubscriptionId id);

    [[nodiscard]] std::uint64_t published_count() const {
        return published_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t dispatched_count() const {
        return dispatched_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t dropped_count() const {
        return dropped_count_.load(std::memory_order_relaxed);
    }

private:
    template <typename T>
    class MpmcBoundedQueue {
    public:
        explicit MpmcBoundedQueue(std::size_t capacity);

        MpmcBoundedQueue(const MpmcBoundedQueue&) = delete;
        MpmcBoundedQueue& operator=(const MpmcBoundedQueue&) = delete;

        [[nodiscard]] bool enqueue(T item);
        [[nodiscard]] bool dequeue(T& item);
        [[nodiscard]] bool empty() const;

    private:
        struct Cell {
            std::atomic<std::size_t> sequence{0};
            T data{};
        };

        static std::size_t round_up_power_of_two(std::size_t value);

        const std::size_t capacity_;
        const std::size_t mask_;
        std::unique_ptr<Cell[]> buffer_;
        alignas(64) std::atomic<std::size_t> enqueue_pos_{0};
        alignas(64) std::atomic<std::size_t> dequeue_pos_{0};
    };

    struct Subscriber {
        SubscriptionId id = 0;
        EventHandler handler;
    };

    struct EventTypeHash {
        std::size_t operator()(EventType type) const {
            return static_cast<std::size_t>(type);
        }
    };

    void worker_loop();
    void dispatch(const IEvent& event);
    void normalize(IEvent& event) const;
    [[nodiscard]] static std::string make_event_id();

    MpmcBoundedQueue<IEvent> queue_;

    std::atomic<bool> running_{false};
    std::thread worker_;
    mutable std::mutex wake_mtx_;
    std::condition_variable wake_cv_;

    std::mutex subscribers_mtx_;
    std::unordered_map<EventType, std::vector<Subscriber>, EventTypeHash> subscribers_;
    std::atomic<SubscriptionId> next_subscription_id_{1};

    std::atomic<std::uint64_t> published_count_{0};
    std::atomic<std::uint64_t> dispatched_count_{0};
    std::atomic<std::uint64_t> dropped_count_{0};
};

} // namespace revia::core

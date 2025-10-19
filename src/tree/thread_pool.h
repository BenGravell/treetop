#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool {
   public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i)
            workers.emplace_back([this] { workerLoop(); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        cond.notify_all();
        for (auto& t : workers) t.join();
    }

    template <typename F>
    auto enqueue(F&& f)
        -> std::future<typename std::result_of<F()>::type> {
        using R = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.emplace([task]() { (*task)(); });
        }
        cond.notify_one();
        return task->get_future();
    }

   private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cond.wait(lock, [&] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cond;
    bool stop = false;
};

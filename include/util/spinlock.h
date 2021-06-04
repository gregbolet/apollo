
#ifndef UTIL_SPINLOCK_H
#define UTIL_SPINLOCK_H

#include <atomic>

namespace util
{
    
class spinlock {
    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;

public:

    spinlock()
        { }

    void lock() {
        while (m_lock.test_and_set(std::memory_order_acquire))
            ;
    }

    void unlock() {
        m_lock.clear(std::memory_order_release);
    }
};

} // namespace

#endif
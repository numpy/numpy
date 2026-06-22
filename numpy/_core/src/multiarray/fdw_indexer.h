template <typename T, typename HashFunc, typename EqualFunc>
class FDWIndexer {
    static constexpr uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
    static constexpr int64_t EMPTY = -1;

    static uint64_t mix(uint64_t x) {
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return x;
    }

    struct Slot {
        T key;
        int64_t value;  // >= 0: unique id; < 0: empty
    };

    std::vector<Slot> slots_;
    size_t mask_;
    size_t size_;
    HashFunc hash_func_;
    EqualFunc equal_func_;

    size_t probe(T key) const {
        uint64_t bits = hash_func_(key);
        return mix(bits * GOLDEN) & mask_;
    }

public:
    explicit FDWIndexer(size_t n, HashFunc hash_func, EqualFunc equal_func)
        : size_(0), hash_func_(hash_func), equal_func_(equal_func) {
        size_t cap = 16;
        while (cap < n * 2) cap <<= 1;
        slots_.resize(cap, Slot{{}, EMPTY});
        mask_ = cap - 1;
    }

    std::pair<int64_t, bool> emplace(T key) {
        size_t idx = probe(key);
        while (true) {
            if (slots_[idx].value < 0) {
                slots_[idx].key = key;
                slots_[idx].value = static_cast<int64_t>(size_);
                size_++;
                return {slots_[idx].value, true};
            }
            if (equal_func_(slots_[idx].key, key))
                return {slots_[idx].value, false};
            idx = (idx + 1) & mask_;
        }
    }

    size_t size() const { return size_; }
};

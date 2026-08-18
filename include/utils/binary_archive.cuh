#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Minimal host-side binary archive for cache / reference files.
//
// Files start with a magic number and a version, so incompatible or truncated
// files are detected eagerly instead of being silently misread.  Each vector is
// written with a byte-length prefix and is resized automatically on read.
class BinaryArchive {
    std::fstream fs;

public:
    static constexpr char MAGIC[4] = { 'L', 'K', 'Z', '1' };
    static constexpr uint32_t VERSION = 1;

    BinaryArchive() = default;
    BinaryArchive(const BinaryArchive&) = delete;
    BinaryArchive& operator=(const BinaryArchive&) = delete;

    ~BinaryArchive() { close(); }

    void open_for_write(const std::string &path) {
        close();
        fs.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!fs) throw std::runtime_error("BinaryArchive: cannot open file for writing: " + path);
        fs.write(MAGIC, sizeof(MAGIC));
        fs.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    }

    void open_for_read(const std::string &path) {
        close();
        fs.open(path, std::ios::binary | std::ios::in);
        if (!fs) throw std::runtime_error("BinaryArchive: cannot open file for reading: " + path);

        char magic[4] = {0, 0, 0, 0};
        uint32_t version = 0;
        fs.read(magic, sizeof(magic));
        fs.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!fs || std::string(magic, sizeof(magic)) != std::string(MAGIC, sizeof(MAGIC)) || version != VERSION)
            throw std::runtime_error("BinaryArchive: incompatible or truncated file: " + path);
    }

    void close() {
        if (fs.is_open()) fs.close();
    }

    bool good() const { return fs.good(); }

    template <typename T>
    void write(const T &value) {
        static_assert(std::is_trivially_copyable<T>::value, "BinaryArchive::write: T must be trivially copyable");
        fs.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    void read(T &value) {
        static_assert(std::is_trivially_copyable<T>::value, "BinaryArchive::read: T must be trivially copyable");
        fs.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!fs) throw std::runtime_error("BinaryArchive: unexpected end of file");
    }

    template <typename T>
    void write(const std::vector<T> &v) {
        static_assert(std::is_trivially_copyable<T>::value, "BinaryArchive::write: element type must be trivially copyable");
        uint64_t bytes = v.size() * sizeof(T);
        fs.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
        fs.write(reinterpret_cast<const char*>(v.data()), bytes);
    }

    template <typename T>
    void read(std::vector<T> &v) {
        static_assert(std::is_trivially_copyable<T>::value, "BinaryArchive::read: element type must be trivially copyable");
        uint64_t bytes = 0;
        fs.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
        if (!fs) throw std::runtime_error("BinaryArchive: unexpected end of file");
        if (bytes % sizeof(T) != 0) throw std::runtime_error("BinaryArchive: corrupt vector length");
        v.resize(bytes / sizeof(T));
        if (bytes > 0) {
            fs.read(reinterpret_cast<char*>(v.data()), bytes);
            if (!fs) throw std::runtime_error("BinaryArchive: truncated vector data");
        }
    }
};

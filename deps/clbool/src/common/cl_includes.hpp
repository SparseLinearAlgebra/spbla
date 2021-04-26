#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <libutils/timer.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include "CL/opencl.hpp"

#define FPGA
#define DEBUG_ENABLE 1
#define DETAIL_DEBUG_ENABLE 1




#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__))
#define WIN
#endif

#if DEBUG_ENABLE
inline std::string LOG_PATH = "../log/log_GPU_merge_path.txt";
#endif

inline std::ostream & get_log_stream(const std::string& path = "") {
    if (path.empty()) {
        return std::cout;
    }
    static std::fstream fstream;
    fstream.open(path, std::fstream::in | std::fstream::out | std::fstream::trunc);
    if (!fstream.is_open()) {
        std::cerr << "cannot open file\n";
    }
    return fstream;
}

// https://stackoverflow.com/a/51802606
struct Logg {
    inline static std::ostream &stream = get_log_stream(/*LOG_PATH*/);

    Logg() {
        stream << "[LOG] ";
    }

    ~Logg() { stream << "\n"; }
};

//inline std::ostream& Log::stream = get_log_stream();

template<typename T>
Logg &&operator<<(Logg &&wrap, T const &whatever) {
    ::std::cout << whatever;
    return ::std::move(wrap);
}


#define SET_TIMER timer t;
#define START_TIMING do { t.restart(); } while(0);
#define END_TIMING(msg) do { t.elapsed(); if constexpr (DEBUG_ENABLE) Logg() << (msg) << t.last_elapsed(); } while(0);
#define LOG if constexpr (DEBUG_ENABLE) Logg()


#pragma once
#include <libutils/timer.h>


#include <string>
#include <iostream>
#include <fstream>

#include "CL/opencl.hpp"
#include "../core/error.hpp"
#define DEBUG_ENABLE 0


#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__))
#define WIN
#endif

#if DEBUG_ENABLE
inline std::string LOG_PATH = "C:/Users/mkarp/GitReps/clean_matrix/sparse_boolean_matrix_operations/log/log.txt";
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
    inline static std::ostream &stream = get_log_stream("");

    Logg() {
        stream << "[LOG] ";
    }

    ~Logg() { stream << "\n"; }
};

//inline std::ostream& Log::stream = get_log_stream();

template<typename T>
Logg &&operator<<(Logg &&wrap, T const &whatever) {
    Logg::stream << whatever;
    return ::std::move(wrap);
}

inline void handle_run(cl_int) {}
inline void handle_run(const cl::Event& e) {
    if constexpr (DEBUG_ENABLE) e.wait();
}
inline void handle_run(void) {}


#define START_TIMING timer t; t.restart();
#define END_TIMING(msg) do { t.elapsed(); if constexpr (DEBUG_ENABLE) Logg() << (msg) << t.last_elapsed(); } while(0);
#define LOG if constexpr (DEBUG_ENABLE) Logg()

#define TIME_RUN(msg, run) {                                             \
        timer t;                                                         \
        t.restart();                                                     \
        handle_run(run);                                                 \
        t.elapsed();                                                     \
        if constexpr (DEBUG_ENABLE) Logg() << (msg) << " " << t.last_elapsed(); \
        }




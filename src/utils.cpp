#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "utils.hpp"

namespace hqp {

Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>& in) {
    /**
     * @brief Identifies and returns the indices of all true elements in a given boolean array.
     *
     * Iterates over the array and collects indices where the condition evaluates to true.
     *
     * @param in Boolean array to be scanned.
     * @return VectorXi containing the positions of true values.
     */
    Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
    for (auto j = 0, i = 0; i < in.rows(); ++i) {
        if (in(i)) out(j++) = i;
    }
    return out;
}



std::string Logger::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
    std::string timeStr(std::ctime(&timeNow));
    // Remove the trailing newline character.
    if (!timeStr.empty() && timeStr.back() == '\n') {
        timeStr.pop_back();
    }
    return timeStr;
}

Logger::Logger(const std::string& filename) : filename_(filename) {
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream ofs(filename_, std::ios::out | std::ios::app);
    if (!ofs) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }
    for (const auto& entry : logBuffer_) {
        ofs << entry << "\n";
    }
    ofs.close();
}

// Log a message by buffering it (with a timestamp).
void Logger::log(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    logBuffer_.push_back(getCurrentTime() + "\n" + message + "\n");
}

}  // namespace hqp

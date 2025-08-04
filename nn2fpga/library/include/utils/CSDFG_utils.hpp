#pragma once
#include <vector>
#include <set>
#include <tuple>
#include <unordered_map>
#include <queue>

class FiringStatus {
public:
  FiringStatus(size_t remaining_time, size_t firing_index)
      : remaining_time(remaining_time), firing_index(firing_index) {}

  bool operator<(const FiringStatus &other) const {
    return std::tie(remaining_time, firing_index) <
           std::tie(other.remaining_time, other.firing_index);
  }

  bool operator==(const FiringStatus &other) const {
    return remaining_time == other.remaining_time &&
           firing_index == other.firing_index;
  }

  bool operator--() {
    --remaining_time;
    return remaining_time == 0;
  }

  size_t get_remaining_time() const { return remaining_time; }
  size_t get_firing_index() const { return firing_index; }

private:
  size_t remaining_time; // Remaining time for the firing to complete.
  size_t firing_index;   // Index of the firing within the actor.
};

#ifndef __SYNTHESIS__
class ActorStatus {
public:
  ActorStatus() : ActorStatus(1, 1) {}

  ActorStatus(size_t t, size_t N) : firings(), current_index(0), t(t), N(N) {
    // Initialize the actor status with the given execution time and number of
    // firings.
  }

  void fire() {
    // Add a new firing to the actor's status.
    firings.insert(FiringStatus(t, current_index));
    current_index = (current_index + 1) % N;
  }

  void advance() {
    // Advance the firings of the actor.
    if (!firings.empty()) {
      std::vector<FiringStatus> updated_firings;
      for (const auto &firing : firings) {
        FiringStatus updated = firing;
        --updated;
        updated_firings.push_back(updated);
      }
      firings.clear();
      for (const auto &firing : updated_firings) {
        if (firing.get_remaining_time() > 0) {
          firings.insert(firing);
        }
      }
    }
  }

  bool empty() const { return firings.empty(); }
  size_t size() const { return firings.size(); }

  bool operator==(const ActorStatus &other) const {
    return firings == other.firings && current_index == other.current_index;
  }

  // Define a string representation for debugging
  std::string to_string() const {
    std::string result = "ActorStatus:\n";
    result += "Current Index: " + std::to_string(current_index) + "\n";
    result += "Firings: ";
    for (const auto &firing : firings) {
      result += "(" + std::to_string(firing.get_remaining_time()) + ", " +
                std::to_string(firing.get_firing_index()) + ") ";
    }
    return result;
  }

  std::multiset<FiringStatus> get_firings() const { return firings; }
  size_t get_current_index() const { return current_index; }

private:
  std::multiset<FiringStatus> firings; // Set of current firings for the actor.
  size_t current_index; // Current index in the actor execution sequence.
  size_t t;             // Execution time of the firing.
  size_t N;             // Number of firings in the execution sequence.
};
#else
class ActorStatus {
public:
  ActorStatus() {}
  ActorStatus(size_t t, size_t N) {
    // Utilize t and N to remove the warning about unused parameters.
    (void)t;
    (void)N;
  }

  void fire() {}
  void advance() {}
  bool empty() const { return true; }
  size_t size() const { return 0; }
  bool operator==(const ActorStatus &other) const {
    (void)other; // Avoid unused parameter warning
    return true;
  }
  std::string to_string() const { return ""; }
  std::multiset<FiringStatus> get_firings() const { return {}; }
  size_t get_current_index() const { return 0; }
};
#endif // __SYNTHESIS__

class CSDFGState {
public:
  CSDFGState() : tokens(), actor_statuses() {}

  CSDFGState(const std::vector<ActorStatus> &actor_statuses,
             const std::vector<size_t> &tokens)
      : tokens(tokens), actor_statuses(actor_statuses) {}

  bool operator==(const CSDFGState &other) const {
    return tokens == other.tokens && actor_statuses == other.actor_statuses;
  }

  std::vector<size_t> get_tokens() const { return tokens; }

  std::vector<ActorStatus> get_actor_statuses() const {
    return actor_statuses;
  }

  void clear() {
    tokens.clear();
    actor_statuses.clear();
  }

  void set_tokens(const std::vector<size_t> &new_tokens) {
    tokens = new_tokens;
  }

  void set_actor_statuses(const std::vector<ActorStatus> &new_actor_statuses) {
    actor_statuses = new_actor_statuses;
  }

  // Define a string representation for debugging
  std::string to_string() const {
    std::string result = "CSDFGState:\n";
    result += "Channel quantity: ";
    for (const auto &token : tokens) {
      result += std::to_string(token) + " ";
    }
    result += "Actor Statuses: \n";
    for (const auto &status : actor_statuses) {
      result += status.to_string() + "\n";
    }
    return result;
  }

private:
  std::vector<size_t> tokens; // Channel quantity. Associate with each channel the
                           // amount of tokens it has.
  std::vector<ActorStatus>
      actor_statuses; // Each actor has a
                      // vector of ActorStatus, which contains the
                      // current firings and the current index.
};

struct CSDFGStateHasher {
  std::size_t operator()(const CSDFGState &s) const {
    std::size_t h = 0;
    for (int t : s.get_tokens()) {
      h ^= std::hash<int>()(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    for (const auto &actor : s.get_actor_statuses()) {
      for (const auto &firing : actor.get_firings()) {
        h ^= std::hash<int>()(firing.get_remaining_time()) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<int>()(firing.get_firing_index()) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
      }
      h ^= std::hash<int>()(actor.get_current_index()) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

#ifndef __SYNTHESIS__
template <typename T> class PipelineDelayBuffer {
public:
  PipelineDelayBuffer(size_t depth) : pipeline_depth(depth) {
    for (size_t i = 0; i < pipeline_depth - 1; ++i) {
      valid_flags.push(false);
    }
  }

  PipelineDelayBuffer() : pipeline_depth(1) {}

  // Push a new output element into the pipeline (valid == true) or a delay slot
  // (valid == false)
  void push(const T &value, bool valid) {
    if (valid) {
      data_queue.push(value);
    }
    valid_flags.push(valid);
  }

  // Step the pipeline forward, and return whether the front is valid and its
  // value
  bool pop(T &out_value) {
    bool valid = valid_flags.front();
    valid_flags.pop();

    if (valid) {
      out_value = data_queue.front();
      data_queue.pop();
    }
    return valid;
  }

  std::string to_string() const {
    std::string result = "PipelineDelayBuffer:\n";
    result += "Depth: " + std::to_string(pipeline_depth) + "\n";
    result += "Data Queue Size: " + std::to_string(data_queue.size()) + "\n";
    result += "Valid Flags: ";
    std::queue<bool> temp_flags = valid_flags;
    while (!temp_flags.empty()) {
      result += (temp_flags.front() ? "1 " : "0 ");
      temp_flags.pop();
    }
    return result;
  }

private:
  size_t pipeline_depth;        // Depth of the pipeline
  std::queue<T> data_queue;     // Queue to hold the data elements
  std::queue<bool> valid_flags; // Queue to hold the valid flags
};
#else
template <typename T> class PipelineDelayBuffer {
public:
  PipelineDelayBuffer() {}
  PipelineDelayBuffer(size_t depth) {
    (void)depth; // Avoid unused parameter warning
  }
  void push(const T &value, bool valid) {
    (void)value; // Avoid unused parameter warning
    (void)valid; // Avoid unused parameter warning
  }

  bool pop(T &out_value) {
    (void)out_value; // Avoid unused parameter warning
    return false;    // Always return false in synthesis mode
  }

  std::string to_string() const { return ""; }
};
#endif // __SYNTHESIS__

#pragma once

#include <string>

/// <summary>
/// Makes a participant ID safe to use in a file name:
///  - leading / trailing whitespace is dropped
///  - each space (or tab) becomes a dash
///  - everything that is not an ASCII letter or digit is removed
/// "  John Smith #3 " -> "John-Smith-3"
/// </summary>
inline std::string sanitizeParticipantId(const std::string& raw)
{
    auto isBlank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };

    size_t begin = 0;
    size_t end = raw.size();
    while (begin < end && isBlank(raw[begin])) ++begin;
    while (end > begin && isBlank(raw[end - 1])) --end;

    std::string out;
    for (size_t i = begin; i < end; ++i) {
        const char c = raw[i];
        if (c == ' ' || c == '\t')
            out.push_back('-');
        else if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
            out.push_back(c);
        // anything else is dropped
    }
    return out;
}

namespace ParticipantPrompt {

    /// <summary>
    /// Shows a small modal popup asking for the participant ID and blocks until it is closed.
    /// The ID is sanitized with sanitizeParticipantId() and is never empty.
    /// </summary>
    /// <param name="participantIdOut">The sanitized ID (only written when this returns true).</param>
    /// <returns>True if the user pressed OK / Enter, false if they cancelled or closed the window.</returns>
    bool ask(std::string& participantIdOut);

}
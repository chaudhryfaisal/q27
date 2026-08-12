#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace q27 {
enum class MetalEndpoint { Completions, Chat, Messages, Responses };

inline uint32_t metal_default_max_tokens(MetalEndpoint endpoint) {
    switch(endpoint) {
        case MetalEndpoint::Completions:
        case MetalEndpoint::Chat: return 256;
        case MetalEndpoint::Messages: return 1024;
        case MetalEndpoint::Responses: return 4096;
    }
    return 256;
}

// Request-local speculation width. Admission must reserve only for the decode
// path this request can actually take, not every server-level capability.
inline uint32_t metal_serving_speculation_width(
    uint32_t mtp_width, bool has_mtp,
    bool chunked_prefill, bool sampled, bool sample_plain,
    bool bounded_reasoning) {
    if (bounded_reasoning || !chunked_prefill) return 0;
    if (mtp_width && has_mtp && !(sampled && sample_plain)) return mtp_width;
    return 0;
}

inline uint32_t metal_max_prompt_tokens(uint32_t context,
                                        uint32_t speculation_width) {
    const uint32_t reserve = 1 + speculation_width;
    return context > reserve ? context - reserve : 1;
}

// Once the output count is clamped to the remaining context, speculative
// decode only needs the lanes this request can actually consume. One-token
// (resident-logit) and zero-token requests need no extra prompt reserve.
inline uint32_t metal_max_prompt_tokens_for_request(
    uint32_t context, uint32_t speculation_width, uint32_t output_tokens) {
    if (output_tokens <= 1) return context;
    const uint32_t reserve = std::min(speculation_width, output_tokens - 1);
    return context > reserve ? context - reserve : 1;
}

inline uint32_t metal_max_generation_tokens(uint64_t prompt_tokens,
                                            uint32_t context) {
    if(prompt_tokens>context) return 0;
    return context-(uint32_t)prompt_tokens+(prompt_tokens?1u:0u);
}

inline bool metal_generation_fits(uint64_t prompt_tokens,uint32_t count,
                                  uint32_t context) {
    return count<=metal_max_generation_tokens(prompt_tokens,context);
}

inline std::string metal_snapshot_head_mask_tag(const uint8_t* masks,size_t count) {
    static constexpr char hex[]="0123456789abcdef";
    std::string tag;
    tag.reserve(count*2);
    for(size_t i=0;i<count;i++) {
        tag.push_back(hex[masks[i]>>4]);
        tag.push_back(hex[masks[i]&15]);
    }
    return tag;
}

inline bool metal_tool_constraint_enabled(bool constrain_tools,bool has_tools,
                                          bool greedy,uint32_t speculation_width,
                                          bool forced_tool_choice) {
    return constrain_tools && has_tools && greedy && speculation_width==0 &&
           !forced_tool_choice;
}

inline bool responses_closed_tool_tail_after_segment(
    bool closed_tool_tail, bool just_closed_tool, bool reasoning,
    const std::string& text) {
    if(reasoning) return false;
    if(just_closed_tool) closed_tool_tail=true;
    if(text.find_first_not_of(" \t\r\n")!=std::string::npos) return false;
    return closed_tool_tail;
}

inline int responses_output_index_after_stream_item(
    int current_index, int reserved_index, bool emitted) {
    return emitted && reserved_index>=0 ? reserved_index+1 : current_index;
}

inline bool responses_token_limit_remains(bool token_limit_reached,
                                          bool accepted_tool_call,
                                          bool tool_calls_clean,
                                          bool final_tool_incomplete) {
    return token_limit_reached &&
           !(accepted_tool_call && tool_calls_clean && !final_tool_incomplete);
}


} // namespace q27

// Replay missed tool-call turns (from toolcall_parity.py --dump) through the
// CURRENT parser, to separate a parser gap from an emission nothing should
// recover. A call truncated mid-value or a <parameter=> with no name is
// correctly refused: executing it would mean inventing content. Only the ones
// that come back with calls are work.
//
//   ./build/replay_missed_calls tools.json misses/miss.*.txt
//
// tools.json is the request's tool schema (an anthropic "tools" array), which
// matters: recovery routinely depends on key inference, and a schema without
// `required` silently disables it.
#include "api_common.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using json = nlohmann::json;
static q27::OrderedToolOutput as_tool(const std::string& b,const json* tools){
  std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> segs;
  segs.emplace_back(q27::StreamSplitter::TOOL,b);
  return q27::resolve_ordered_tool_segments(segs,tools,false,[](const std::string&,size_t){return true;});
}
int main(int argc,char** argv){
  if(argc<3){ fprintf(stderr,"usage: %s tools.json miss.*.txt\n",argv[0]); return 2; }
  std::ifstream tf(argv[1]); std::stringstream ts; ts<<tf.rdbuf();
  json tools=json::parse(ts.str());
  int text_ok=0, tool_ok=0, dead=0;
  for(int i=2;i<argc;i++){
    std::ifstream f(argv[i]); std::stringstream ss; ss<<f.rdbuf();
    std::string s=ss.str();
    size_t r=s.find("\n===\n"); std::string body = r==std::string::npos? s : s.substr(r+5);
    std::string pre; auto tv=q27::parse_bare_tool_calls(body,&pre,&tools);
    auto ov=as_tool(body,&tools);
    const char* name=strrchr(argv[i],'/'); name=name?name+1:argv[i];
    bool ok = !tv.empty() || !ov.calls.empty();
    if(!tv.empty()) text_ok++;
    if(!ov.calls.empty()) tool_ok++;
    if(!ok) dead++;
    printf("  %-16s TEXT=%zu TOOL=%zu %s\n", name, tv.size(), ov.calls.size(), ok?"":"  <-- STILL DEAD");
  }
  printf("\n  recovered via TEXT chain: %d   via TOOL chain: %d   still dead: %d\n", text_ok, tool_ok, dead);
  return 0;
}

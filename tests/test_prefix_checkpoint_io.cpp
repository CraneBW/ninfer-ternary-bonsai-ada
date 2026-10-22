// Real file transport regressions: no CUDA/Engine/model is needed.
#include "kvmem/prefix_checkpoint_file.h"
#include "kvmem/kvmem_cold_store.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
using namespace ninfer::kvmem;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::vector<unsigned char> bytes(const std::filesystem::path& p){
 std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};
}
void flip(const std::filesystem::path& p,std::size_t offset){
 std::fstream f(p,std::ios::binary|std::ios::in|std::ios::out);f.seekg(offset);char c=0;f.read(&c,1);c^=1;f.seekp(offset);f.write(&c,1);f.flush();
}
void consume(PrefixCheckpointReader& file,const std::vector<unsigned char>& expected){
 auto& s=file.stream();std::size_t offset=0;const std::size_t pieces[]={1,3,7,63,1024,65537,1048576,35};
 std::size_t i=0;std::vector<unsigned char> result(expected.size());
 while(s.remaining()){const auto n=std::min<std::uint64_t>(pieces[i++%8],s.remaining());s.read(result.data()+offset,n);offset+=n;}
 file.finish();require(result==expected,"complete payload differs across fragmented reads");
}
template<class F> void rejected(F action,const char* message){bool ok=false;try{action();}catch(const SnapshotCorrupt&){ok=true;}require(ok,message);}
}
int main(){
 auto folder=std::filesystem::temp_directory_path()/("ninfer-checkpoint-io-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 std::filesystem::create_directory(folder);auto path=folder/"prefix.nkv";auto index=path.string()+".xxh3idx";
 try{
  std::vector<unsigned char> payload(3*snapshot_chunk+37);std::uint64_t x=1;
  for(auto& v:payload){x=x*6364136223846793005ULL+1;v=static_cast<unsigned char>(x>>56);}
  {PrefixCheckpointWriter f(path.string());f.stream().write(payload.data(),payload.size());f.finish();}
  const auto original=bytes(path);
  {PrefixCheckpointReader f(path.string(),false);consume(f,payload);}
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  require(std::filesystem::exists(index),"legacy-validated index not persisted");const auto original_index=bytes(index);
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  require(bytes(path)==original,"index construction changed checkpoint payload");
  // Invalid index body, truncated index and wrong binding rebuild from FNV.
  flip(index,64);
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  require(bytes(index)==original_index,"invalid digest index was not regenerated");
  std::filesystem::resize_file(index,17);
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  flip(index,32);
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  require(bytes(index)==original_index,"stale binding was not regenerated");
  // Existing valid index cannot hide corruption in the final partial chunk.
  flip(path,payload.size()-1);
  rejected([&]{PrefixCheckpointReader f(path.string(),true);},"indexed payload corruption admitted before mutation");
  rejected([&]{PrefixCheckpointReader f(path.string(),false);},"legacy payload corruption admitted");
  std::filesystem::remove(index);
  rejected([&]{PrefixCheckpointReader f(path.string(),true);},"missing-index path admitted bad FNV");
  require(!std::filesystem::exists(index),"index published for corrupt archive");flip(path,payload.size()-1);
  {PrefixCheckpointReader f(path.string(),true);consume(f,payload);}
  // Full prevalidation cannot license a later changed block during consumption.
  {PrefixCheckpointReader f(path.string(),true);flip(path,snapshot_chunk+23);
   rejected([&]{consume(f,payload);},"during-restore mutation was not detected");}
  flip(path,snapshot_chunk+23);
  // A footer or physical truncation is still rejected before native mutation.
  flip(path,payload.size());
  rejected([&]{PrefixCheckpointReader f(path.string(),true);},"invalid footer admitted");flip(path,payload.size());
  std::filesystem::resize_file(path,original.size()-1);
  rejected([&]{PrefixCheckpointReader f(path.string(),true);},"truncated archive admitted");
#if defined(__linux__)
  // A stable private clone is independent of the original inode and is retained
  // by cold extents after the Reader closes. Freed slots write internal storage.
  auto stable_path=folder/"stable.nkv";
  {PrefixCheckpointWriter f(stable_path.string());f.stream().write(payload.data(),payload.size());f.finish();}
  ::setenv("NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE","1",1);
  ColdKVStore cold(256,folder.string());ColdKVStore::Handle first,second;
  bool cloned=false;
  {PrefixCheckpointReader f(stable_path.string(),true);cloned=f.private_backing();
   if(cloned){
    auto backing=f.stream().stable_backing();
    first=cold.adopt_snapshot_extent(backing,13);second=cold.adopt_snapshot_extent(backing,269);
    flip(stable_path,13); // Only the user's original changes, never the clone.
   }
   consume(f,payload);
  }
  if(cloned){
   std::filesystem::remove(stable_path);
   std::vector<std::byte> a(256),b(256),replacement(256,std::byte{0x3c});
   cold.load(first,a);require(std::memcmp(a.data(),payload.data()+13,256)==0,"private clone lost original bytes");
   auto read=cold.prefetch(second,b);
   require(cold.release(first),"stable extent release failed");auto next=cold.store(replacement);
   require(!cold.valid(first)&&next.generation!=first.generation,"stable slot generation reused");
   // Other owners may grow the directory while this immutable extent is read.
   for(int i=0;i<64;++i) (void)cold.store(replacement);
   read.get();require(std::memcmp(b.data(),payload.data()+269,256)==0,"stable prefetch differs during directory growth");
   cold.load(next,a);require(a==replacement,"reused extent did not write internal storage");
   cold.load(second,b);require(std::memcmp(b.data(),payload.data()+269,256)==0,"reused slot overwrote adjacent stable extent");
   auto corrupt=folder/"private-corrupt.nkv";
   {PrefixCheckpointWriter f(corrupt.string());f.stream().write(payload.data(),payload.size());f.finish();}
   {PrefixCheckpointReader f(corrupt.string(),true);auto backing=f.stream().stable_backing();
    require(bool(backing),"second private clone unavailable");const unsigned char changed=payload[42]^1;
    require(::pwrite(backing->fd,&changed,1,42)==1,"private corruption injection failed");
    rejected([&]{consume(f,payload);},"private source changed during restore was admitted");}
  }
  ::setenv("NINFER_BENCH_CHECKPOINT_LOAD_PIPELINE","0",1);
  std::cout<<"PASS private_clone="<<cloned<<" original-mutation/unlink/reader-close/prefetch/generation/slot-reuse/private-corrupt\n";
#endif
  std::filesystem::remove_all(folder);
  std::cout<<"PASS legacy/build/hit/fragmented/tail/index-invalid/index-stale/corrupt-before/corrupt-during/truncated\n";
 }catch(...){std::filesystem::remove_all(folder);throw;}
}

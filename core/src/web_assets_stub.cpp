// Används när frontend inte är byggd – servern visar då en instruktionssida
// (eller serverar från --web DIR).
#include "web_assets.hpp"

namespace web {

const Asset kAssets[1] = {{nullptr, nullptr, nullptr, 0}};
const std::size_t kAssetCount = 0;

}  // namespace web

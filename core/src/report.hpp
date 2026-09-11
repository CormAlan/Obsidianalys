// JSON-serialisering av analysresultat.
#pragma once

#include <string>

#include "analysis.hpp"

namespace report {

// Hela skanningen. full=true lägger till rubriker och utlänkar per anteckning (för --json).
std::string scan_json(const analysis::Vault& vault, bool full);

// Detaljer för en anteckning: absolut sökväg, rubriker, in- och utlänkar.
std::string note_json(const analysis::Vault& vault, int id);

}  // namespace report

// Random games on KataGo's own rules engine (BoardHistory, cpp/game/boardhistory.cpp), recording its state after every
// move, as goldens for the port (packages/engine-webgpu/src/rules/history.ts). Built against KataGo's source by
// scripts/capture-history-goldens.mjs; not part of any package build.
//
// Reads one JSON game spec per line on stdin; writes one JSON game per line on stdout: the spec, the moves played and
// the states. Moves are chosen with KataGo's Rand seeded from the spec's seed: a pass with probability passProb, else a
// ko capture or pass for ko with probability koBias (if there is one), else a random legal move. With probability
// tolerantProb a move is played with makeBoardMoveTolerant instead (and then, 30% of the time, by the player not to
// move); with probability preventEncoreProb it's played with preventEncore.
#include "game/boardhistory.h"
#include "core/rand.h"
#include "external/nlohmann_json/json.hpp"
#include <iostream>
#include <string>

using json = nlohmann::json;
using namespace std;

static json locJson(Loc loc, const Board& b) {
  if(loc == Board::PASS_LOC) return "pass";
  return json::array({Location::getX(loc, b.x_size), Location::getY(loc, b.x_size)});
}

static string colorsString(const Color* c, const Board& b) {
  string s;
  for(int y = 0; y < b.y_size; y++)
    for(int x = 0; x < b.x_size; x++)
      s += (char)('0' + c[Location::getLoc(x, y, b.x_size)]);
  return s;
}

static string flagsString(const bool* f, const Board& b) {
  string s;
  for(int y = 0; y < b.y_size; y++)
    for(int x = 0; x < b.x_size; x++)
      s += f[Location::getLoc(x, y, b.x_size)] ? '1' : '0';
  return s;
}

static json state(const BoardHistory& hist, const Board& board) {
  json j;
  Player pla = hist.presumedNextMovePla;
  j["colors"] = colorsString(board.colors, board);
  j["koLoc"] = board.ko_loc == Board::NULL_LOC ? json(nullptr) : locJson(board.ko_loc, board);
  j["encorePhase"] = hist.encorePhase;
  j["isGameFinished"] = hist.isGameFinished;
  j["isPastNormalPhaseEnd"] = hist.isPastNormalPhaseEnd;
  j["isScored"] = hist.isScored;
  j["isNoResult"] = hist.isNoResult;
  j["winner"] = (int)hist.winner;
  j["finalWhiteMinusBlackScore"] = hist.finalWhiteMinusBlackScore;
  j["whiteBonusScore"] = hist.whiteBonusScore;
  j["whiteHandicapBonusScore"] = hist.whiteHandicapBonusScore;
  j["consecutiveEndingPasses"] = hist.consecutiveEndingPasses;
  j["numApproxValidTurnsThisPhase"] = hist.numApproxValidTurnsThisPhase;
  j["koRecapBlocked"] = flagsString(hist.koRecapBlocked, board);
  j["superKoBanned"] = flagsString(hist.superKoBanned, board);
  j["secondEncoreStartColors"] = colorsString(hist.secondEncoreStartColors, board);
  j["nextPla"] = (int)pla;
  j["passWouldEndPhase"] = hist.passWouldEndPhase(board, pla);
  j["passWouldEndGame"] = hist.passWouldEndGame(board, pla);
  j["isFinalPhase"] = hist.isFinalPhase();
  // Legal and pass-for-ko for the presumed next player, every on-board point.
  string legal, passForKo;
  for(int y = 0; y < board.y_size; y++)
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      legal += hist.isLegal(board, loc, pla) ? '1' : '0';
      passForKo += hist.isPassForKo(board, loc, pla) ? '1' : '0';
    }
  j["legal"] = legal;
  j["passForKo"] = passForKo;
  // The score and area if the game were scored now.
  {
    BoardHistory copy(hist);
    Color area[Board::MAX_ARR_SIZE];
    copy.endAndScoreGameNow(board, area);
    j["scoreNow"] = copy.finalWhiteMinusBlackScore;
    j["areaNow"] = colorsString(area, board);
  }
  {
    Color area[Board::MAX_ARR_SIZE];
    hist.getAreaNow(board, area);
    j["getAreaNow"] = colorsString(area, board);
  }
  // Independent life area, every combination of its flags.
  json il = json::array();
  for(int keepTerritories = 0; keepTerritories < 2; keepTerritories++)
    for(int keepStones = 0; keepStones < 2; keepStones++)
      for(int excl = 0; excl < 2; excl++) {
        Color area[Board::MAX_ARR_SIZE];
        int count = 0;
        board.calculateIndependentLifeArea(area, count, keepTerritories, keepStones, excl, hist.suicideLegalForPassAlive());
        il.push_back(json::object({{"area", colorsString(area, board)}, {"count", count}}));
      }
  j["independentLife"] = il;
  return j;
}

int main() {
  Board::initHash();
  string line;
  while(getline(cin, line)) {
    if(line.empty()) continue;
    json spec = json::parse(line);
    int xSize = spec["xSize"], ySize = spec["ySize"];
    Rules rules;
    rules.koRule = spec["koRule"];
    rules.scoringRule = spec["scoringRule"];
    rules.taxRule = spec["taxRule"];
    rules.multiStoneSuicideLegal = spec["multiStoneSuicideLegal"];
    rules.hasButton = false;
    rules.whiteHandicapBonusRule = spec["whiteHandicapBonusRule"];
    rules.friendlyPassOk = false;
    rules.komi = spec["komi"];
    Board board(xSize, ySize);
    string init = spec["initial"];
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++) {
        char c = init[y * xSize + x];
        if(c == '1' || c == '2') {
          Loc loc = Location::getLoc(x, y, xSize);
          // Skip stones that would have no liberties.
          board.setStoneFailIfNoLibs(loc, (Color)(c - '0'));
        }
      }
    Player pla = spec["nextPla"];
    int encorePhase = spec["encorePhase"];
    BoardHistoryModes modes(spec["alwaysPassAlive"], spec["excludeAdjAtari"]);
    BoardHistory hist(board, pla, rules, encorePhase, modes);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(spec["assumeHandicap"]);
    Rand rand(spec["seed"].get<string>());
    double passProb = spec["passProb"];
    double preventEncoreProb = spec["preventEncoreProb"];
    double tolerantProb = spec["tolerantProb"];
    double koBias = spec.contains("koBias") ? spec["koBias"].get<double>() : 0.0;
    int numMoves = spec["numMoves"];
    int movesAfterEnd = spec["movesAfterEnd"];

    json game = spec;
    game["initialActual"] = colorsString(board.colors, board);
    game["states"] = json::array();
    game["states"].push_back(state(hist, board));
    json moves = json::array();
    int afterEnd = 0;
    for(int i = 0; i < numMoves; i++) {
      if(hist.isGameFinished) {
        if(afterEnd >= movesAfterEnd) break;
        afterEnd++;
      }
      Player movePla = hist.presumedNextMovePla;
      bool tolerant = rand.nextDouble() < tolerantProb;
      if(tolerant && rand.nextDouble() < 0.3) movePla = getOpp(movePla);
      vector<Loc> candidates;
      for(int y = 0; y < ySize; y++)
        for(int x = 0; x < xSize; x++) {
          Loc loc = Location::getLoc(x, y, xSize);
          bool ok = tolerant ? hist.isLegalTolerant(board, loc, movePla) : hist.isLegal(board, loc, movePla);
          if(ok) candidates.push_back(loc);
        }
      // Ko moves: ko captures and passes for ko, preferred with probability koBias, to make ko fights and cycles.
      vector<Loc> koMoves;
      for(Loc c : candidates)
        if(board.getKoCaptureLoc(c, movePla) != Board::NULL_LOC || hist.isPassForKo(board, c, movePla))
          koMoves.push_back(c);
      Loc loc = Board::PASS_LOC;
      if(!koMoves.empty() && rand.nextDouble() < koBias)
        loc = koMoves[rand.nextUInt((uint32_t)koMoves.size())];
      else if(!candidates.empty() && rand.nextDouble() >= passProb)
        loc = candidates[rand.nextUInt((uint32_t)candidates.size())];
      bool preventEncore = rand.nextDouble() < preventEncoreProb;
      if(tolerant) {
        bool suc = hist.makeBoardMoveTolerant(board, loc, movePla, preventEncore);
        if(!suc) { cerr << "tolerant move failed" << endl; return 1; }
      }
      else
        hist.makeBoardMoveAssumeLegal(board, loc, movePla, NULL, preventEncore);
      moves.push_back(json::object({{"loc", locJson(loc, board)}, {"pla", (int)movePla}, {"preventEncore", preventEncore}, {"tolerant", tolerant}}));
      game["states"].push_back(state(hist, board));
    }
    game["moves"] = moves;
    cout << game.dump() << "\n";
  }
  return 0;
}

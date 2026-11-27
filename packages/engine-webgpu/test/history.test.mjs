// Golden test: the rules engine (GameHistory) against KataGo's own (BoardHistory, KataGo 1.18.2), move by move. The
// goldens (scripts/capture-history-goldens.mjs) are random games played on KataGo's rules engine, with a digest of its
// state after every move: the board, the encore phase and its ko recapture blocks and once-only bans, superko bans,
// legality and passes for ko for every point, pass endings, the end of the game and its score, the score and area if the
// game were scored now, and the independent life area. Area and territory scoring, every tax rule and ko rule, the
// encore, the encore prevented, no-result cycles, tolerant moves and moves out of turn, as the games happen to reach
// them; the test counts what they reached. A second test checks the network's inputs for the player to move at every
// state (encodeHistoryV7 against KataGo's NNInputs::fillRowV7), exactly, with each game's input options. A third checks
// the search's position keys (situationKey, graphStateKey): across all states on boards of the same size, two states' keys
// are equal exactly when KataGo's hashes are (getSituationRulesAndKoHash, GraphHash::getStateHash).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  Board, GameHistory, NULL_LOC, PASS_LOC, calculateIndependentLifeArea, encodeHistoryV7, graphStateKey, situationKey,
} from '../dist/index.js';
import { digestInputs, digestState } from './history-state.mjs';

const goldens = JSON.parse(readFileSync(new URL('./goldens/history-katago.json', import.meta.url), 'utf8'));

const KO = ['SIMPLE', 'POSITIONAL', 'SITUATIONAL'];
const SCORING = ['AREA', 'TERRITORY'];
const TAX = ['NONE', 'SEKI', 'ALL'];
const WHB = ['0', 'N', 'N-1'];

/** One character per point, row by row from the top. */
function points(board, valueAt) {
  let s = '';
  for (let y = 0; y < board.ySize; y++) for (let x = 0; x < board.xSize; x++) s += valueAt(board.loc(x, y));
  return s;
}

/** The recorded state of a GameHistory, as scripts/native/history-dump.cpp records KataGo's. */
function historyState(h) {
  const b = h.board;
  const pla = h.nextPla;
  const scored = h.clone();
  const area = scored.endAndScoreGameNow();
  const getAreaNow = h.getAreaNow();
  const independentLife = [];
  for (const keepTerritories of [false, true]) {
    for (const keepStones of [false, true]) {
      for (const exclude of [false, true]) {
        const r = calculateIndependentLifeArea(b, keepTerritories, keepStones, exclude, h.suicideLegalForPassAlive());
        independentLife.push({ area: points(b, (l) => r.area[l]), count: r.whiteMinusBlackIndependentLifeRegionCount });
      }
    }
  }
  return {
    colors: points(b, (l) => b.colors[l]),
    koLoc: b.koLoc === NULL_LOC ? null : [b.x(b.koLoc), b.y(b.koLoc)],
    encorePhase: h.encorePhase,
    isGameFinished: h.isGameFinished,
    isPastNormalPhaseEnd: h.isPastNormalPhaseEnd,
    isScored: h.isScored,
    isNoResult: h.isNoResult,
    winner: h.winner ?? 0,
    finalWhiteMinusBlackScore: h.finalWhiteMinusBlackScore,
    whiteBonusScore: h.whiteBonusScore,
    whiteHandicapBonusScore: h.whiteHandicapBonusScore,
    consecutiveEndingPasses: h.consecutiveEndingPasses,
    numApproxValidTurnsThisPhase: h.numApproxValidTurnsThisPhase,
    koRecapBlocked: points(b, (l) => (h.koRecapBlocked[l] ? 1 : 0)),
    superKoBanned: points(b, (l) => (h.superKoBanned[l] ? 1 : 0)),
    secondEncoreStartColors: points(b, (l) => h.secondEncoreStartColors[l]),
    nextPla: pla,
    passWouldEndPhase: h.passWouldEndPhase(pla),
    passWouldEndGame: h.passWouldEndGame(pla),
    isFinalPhase: h.isFinalPhase(),
    legal: points(b, (l) => (h.isLegal(l, pla) ? 1 : 0)),
    passForKo: points(b, (l) => (h.isPassForKo(l, pla) ? 1 : 0)),
    scoreNow: scored.finalWhiteMinusBlackScore,
    areaNow: points(b, (l) => area[l]),
    getAreaNow: points(b, (l) => getAreaNow[l]),
    independentLife,
  };
}

/** Replays a golden game on GameHistory, calling `onState(history, i)` at the start and after each move. */
function replay(g, gi, onState) {
  const board = new Board(g.xSize, g.ySize);
  for (let y = 0; y < g.ySize; y++) {
    for (let x = 0; x < g.xSize; x++) {
      const c = g.initial[y * g.xSize + x];
      if (c === '1' || c === '2') board.setStoneFailIfNoLibs(board.loc(x, y), Number(c));
    }
  }
  const rules = {
    koRule: KO[g.koRule], scoringRule: SCORING[g.scoringRule], taxRule: TAX[g.taxRule], multiStoneSuicideLegal: g.multiStoneSuicideLegal,
    hasButton: false, whiteHandicapBonusRule: WHB[g.whiteHandicapBonusRule], friendlyPassOk: false,
  };
  const modes = { alwaysComputePassAliveUnderSuicideRules: g.alwaysPassAlive, excludeTerritoryAdjacentToAtari: g.excludeAdjAtari };
  const h = new GameHistory(board, g.nextPla, rules, g.komi, g.encorePhase, modes);
  h.setAssumeMultipleStartingBlackMovesAreHandicap(g.assumeHandicap);
  onState(h, 0, null);
  g.moves.forEach(([x, y, pla, flags], i) => {
    const loc = x < 0 ? PASS_LOC : board.loc(x, y);
    const preventEncore = (flags & 1) !== 0;
    const wasPassForKo = loc !== PASS_LOC && h.isPassForKo(loc, pla);
    if (flags & 2) assert.ok(h.playTolerant(loc, pla, preventEncore), `game ${gi}, move ${i}: KataGo played it`);
    else h.play(loc, pla, preventEncore);
    onState(h, i + 1, { wasPassForKo });
  });
}

test('the rules engine matches KataGo 1.18.2 move by move', () => {
  const reached = { states: 0, encore1: 0, encore2: 0, recapBlocks: 0, passesForKo: 0, encoreBans: 0, superkoBans: 0, noResults: 0, encorePrevented: 0, scored: 0 };
  for (const [gi, g] of goldens.games.entries()) {
    replay(g, gi, (h, i, move) => {
      const state = historyState(h);
      assert.equal(digestState(state), g.digests.slice(16 * i, 16 * i + 16), `game ${gi}, state ${i} (after ${i} moves) differs from KataGo's; rerun the capture with --full to see KataGo's`);
      reached.states++;
      if (move?.wasPassForKo) reached.passesForKo++;
      if (state.encorePhase === 1) reached.encore1++;
      if (state.encorePhase === 2) reached.encore2++;
      if (state.koRecapBlocked.includes('1')) reached.recapBlocks++;
      if (state.encorePhase > 0 && state.superKoBanned.includes('1')) reached.encoreBans++;
      if (state.encorePhase === 0 && state.superKoBanned.includes('1')) reached.superkoBans++;
      if (state.isNoResult) reached.noResults++;
      if (state.isPastNormalPhaseEnd) reached.encorePrevented++;
      if (state.isScored) reached.scored++;
    });
  }
  // What the games reached, so a recapture that reaches less fails here rather than passing on less.
  assert.ok(reached.encore1 > 1000 && reached.encore2 > 1000, `encore states: ${reached.encore1}, ${reached.encore2}`);
  for (const k of ['recapBlocks', 'passesForKo', 'encoreBans', 'superkoBans', 'encorePrevented', 'scored']) {
    assert.ok(reached[k] >= 20, `${k}: ${reached[k]}`);
  }
  assert.ok(reached.noResults >= 3, `no-results: ${reached.noResults}`);
});

test("the network's inputs match KataGo 1.18.2 at every state", () => {
  let states = 0;
  for (const [gi, g] of goldens.games.entries()) {
    const p = g.inputParams;
    const options = {
      drawEquivalentWinsForWhite: p.drawEquivalentWinsForWhite, conservativePassAndIsRoot: p.conservativePassAndIsRoot,
      enablePassingHacks: p.enablePassingHacks, maxHistory: p.maxHistory,
    };
    replay(g, gi, (h, i) => {
      const { spatial, global } = encodeHistoryV7(h, options).input;
      const inputs = { spatial: Array.from(spatial, (v) => (v !== 0 ? '1' : '0')).join(''), global: Array.from(global) };
      assert.equal(digestInputs(inputs), g.inputDigests.slice(16 * i, 16 * i + 16), `game ${gi}, state ${i}: inputs differ from KataGo's; rerun the capture with --full to see KataGo's`);
      states++;
    });
  }
  assert.equal(states, goldens.games.reduce((n, g) => n + g.moves.length + 1, 0));
});

/** [game, state, first game, first state] for each key equal to an earlier one on a board of the same size (as the capture
 * records KataGo's hashes). */
function repeats(games, keys) {
  const first = new Map();
  const out = [];
  keys.forEach((gameKeys, gi) => {
    gameKeys.forEach((k, si) => {
      const v = `${games[gi].xSize}x${games[gi].ySize}|${k}`;
      const f = first.get(v);
      if (f) out.push([gi, si, ...f]);
      else first.set(v, [gi, si]);
    });
  });
  return out;
}

test("the search's position keys tell states apart exactly as KataGo's hashes do", () => {
  const situation = [];
  const graphState = [];
  for (const [gi, g] of goldens.games.entries()) {
    const draw = g.inputParams.drawEquivalentWinsForWhite;
    const s = [];
    const t = [];
    replay(g, gi, (h) => {
      s.push(situationKey(h, h.nextPla, draw));
      t.push(graphStateKey(h, h.nextPla, draw));
    });
    situation.push(s);
    graphState.push(t);
  }
  const first = (list) => list.slice(0, 5).map((r) => `game ${r[0]} state ${r[1]} = game ${r[2]} state ${r[3]}`).join('; ');
  const diff = (ours, katago) => {
    const k = new Set(katago.map((r) => r.join()));
    const o = new Set(ours.map((r) => r.join()));
    return `only ours: ${first(ours.filter((r) => !k.has(r.join())))}; only KataGo's: ${first(katago.filter((r) => !o.has(r.join())))}`;
  };
  const sit = repeats(goldens.games, situation);
  assert.ok(JSON.stringify(sit) === JSON.stringify(goldens.situationRepeats), `situation keys: ${diff(sit, goldens.situationRepeats)}`);
  const graph = repeats(goldens.games, graphState);
  assert.ok(JSON.stringify(graph) === JSON.stringify(goldens.graphStateRepeats), `graph-search state keys: ${diff(graph, goldens.graphStateRepeats)}`);
  assert.ok(goldens.situationRepeats.length >= 500, `repeated situations: ${goldens.situationRepeats.length}`);
});

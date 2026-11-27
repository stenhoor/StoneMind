// The query fields of part B, step 11c, below the search: the position KataGo's analysis engine builds
// (src/rules/analysis-position.ts), and the avoid lists and config loading of src/search/query-settings.ts. The search
// with each field is checked against native KataGo in test/search.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  Board, BLACK, WHITE, EMPTY, PASS_LOC, NULL_LOC, parseRules, analysisHistory, avoidMoveUntilByLoc, searchParamsFromConfig,
  DEFAULT_SEARCH_PARAMS, ConfigValueError, UnsupportedSettingError, IllegalMoveError,
} from '../dist/index.js';

const blank = new Board(9, 9);
const at = (v) => blank.loc('ABCDEFGHJ'.indexOf(v[0]), 9 - Number(v.slice(1)));
const position = (fields) => ({ xSize: 9, ySize: 9, rules: parseRules('tromp-taylor'), komi: 7.5, moves: [], ...fields });

test('initial stones capture as KataGo setStone does, and a lone stone with no liberties is left out', () => {
  // White A2 and B1 surround A1; Black A1 would have no liberties and captures nothing, so it's left out.
  const h = analysisHistory(position({ initialStones: [[WHITE, at('A2')], [WHITE, at('B1')], [BLACK, at('A1')]] }));
  assert.equal(h.board.colors[at('A1')], EMPTY);
  // White A1, then Black A2 and B1: Black B1 takes White A1's last liberty and captures it.
  const h2 = analysisHistory(position({ initialStones: [[WHITE, at('A1')], [BLACK, at('A2')], [BLACK, at('B1')]] }));
  assert.equal(h2.board.colors[at('A1')], EMPTY);
  assert.equal(h2.board.colors[at('B1')], BLACK);
  assert.equal(h2.board.koLoc, NULL_LOC);
});

test('the first player: initialPlayer, else the first move\'s, else White with handicap stones', () => {
  assert.equal(analysisHistory(position({})).nextPla, BLACK);
  assert.equal(analysisHistory(position({ initialStones: [[BLACK, at('C3')]] })).nextPla, BLACK);
  assert.equal(analysisHistory(position({ initialStones: [[BLACK, at('C3')], [BLACK, at('G7')]] })).nextPla, WHITE);
  assert.equal(analysisHistory(position({ initialStones: [[BLACK, at('C3')], [BLACK, at('G7')], [WHITE, at('E5')]] })).nextPla, BLACK);
  assert.equal(analysisHistory(position({ initialPlayer: WHITE })).nextPla, WHITE);
  assert.equal(analysisHistory(position({ moves: [[WHITE, at('E5')]] })).moves[0].pla, WHITE);
});

test('a player moving twice starts the history again from that position', () => {
  const h = analysisHistory(position({ moves: [[BLACK, at('C3')], [BLACK, at('G7')], [WHITE, at('E5')]] }));
  assert.equal(h.moves.length, 2);
  assert.equal(h.initialBoard.colors[at('C3')], BLACK);
  assert.equal(h.nextPla, BLACK);
  // turnNumber stops early.
  assert.equal(analysisHistory(position({ moves: [[BLACK, at('C3')], [WHITE, at('G7')]] }), 1).moves.length, 1);
});

test('moves are replayed tolerantly: a ko retaken at once is played', () => {
  const ko = [['E6', BLACK], ['F6', WHITE], ['D5', BLACK], ['G5', WHITE], ['E4', BLACK], ['F4', WHITE], ['A1', BLACK], ['E5', WHITE], ['F5', BLACK], ['E5', WHITE]];
  const h = analysisHistory(position({ moves: ko.map(([v, p]) => [p, at(v)]) }));
  assert.equal(h.board.colors[at('E5')], WHITE);
  assert.equal(h.board.colors[at('F5')], EMPTY);
  assert.equal(h.numApproxValidTurnsThisPhase, 10);
});

test('a move on an occupied point is refused with KataGo\'s message', () => {
  assert.throws(() => analysisHistory(position({ moves: [[BLACK, at('C3')], [WHITE, at('C3')]] })), (e) => e instanceof IllegalMoveError && e.message === 'Illegal move 1: C3');
});

test('avoidMoves and allowMoves lists', () => {
  const size = blank.arraySize;
  const avoid = avoidMoveUntilByLoc(size, [{ player: BLACK, moves: [at('C3'), PASS_LOC], untilDepth: 2 }, { player: BLACK, moves: [at('C3')], untilDepth: 5 }], false);
  assert.equal(avoid.white, null);
  assert.equal(avoid.black[at('C3')], 5);
  assert.equal(avoid.black[PASS_LOC], 2);
  assert.equal(avoid.black[at('D4')], 0);
  const allow = avoidMoveUntilByLoc(size, [{ player: WHITE, moves: [at('C3')], untilDepth: 3 }], true);
  assert.equal(allow.black, null);
  assert.equal(allow.white[at('C3')], 0);
  assert.equal(allow.white[at('D4')], 3);
  assert.equal(allow.white[PASS_LOC], 3);
  assert.throws(
    () => avoidMoveUntilByLoc(size, [{ player: WHITE, moves: [], untilDepth: 1 }, { player: WHITE, moves: [], untilDepth: 1 }], true),
    /Cannot specify allowMoves more than once for the same player/,
  );
});

test('an empty config gives the analysis engine\'s defaults', () => {
  const { params, unusedKeys, maxVisits } = searchParamsFromConfig({});
  assert.deepEqual(params, DEFAULT_SEARCH_PARAMS);
  assert.deepEqual(unusedKeys, []);
  assert.equal(maxVisits, undefined);
});

test('config values are read as KataGo reads them, with its dependent defaults', () => {
  const { params, unusedKeys, maxVisits } = searchParamsFromConfig(
    { cpuctExploration: ' 1.5 ', useLcbForSelection: 'FALSE', rootPolicyTemperature: '2', fpuLossProp: '0.25', policyOptimism: '0.1', maxVisits: '50', analysisPVLen: '3' },
    { batchSize: 4 },
  );
  assert.equal(params.cpuctExploration, 1.5);
  assert.equal(params.useLcbForSelection, false);
  assert.equal(params.rootPolicyTemperatureEarly, 2);
  assert.equal(params.rootFpuLossProp, 0.25);
  assert.equal(params.rootPolicyOptimism, 0.1);
  assert.equal(params.batchSize, 4);
  assert.equal(maxVisits, 50);
  assert.deepEqual(unusedKeys, ['analysisPVLen']);
});

test('config values KataGo refuses, and settings for search features not ported', () => {
  assert.throws(() => searchParamsFromConfig({ cpuctExploration: '11' }), (e) => e instanceof ConfigValueError && /must be in the range 0 to 10/.test(e.message));
  assert.throws(() => searchParamsFromConfig({ cpuctExploration: '1.5x' }), ConfigValueError);
  assert.throws(() => searchParamsFromConfig({ useLcbForSelection: '1' }), ConfigValueError);
  assert.throws(() => searchParamsFromConfig({ graphSearchRepBound: '4.5' }), ConfigValueError);
  assert.throws(() => searchParamsFromConfig({ rootNoiseEnabled: 'true', maxTime: '5' }), (e) => e instanceof UnsupportedSettingError && e.keys.join() === 'rootNoiseEnabled,maxTime');
});

// The digest of a game history's state that the rules goldens record after every move (test/history.test.mjs). Shared
// by the test, which computes the state from the port's GameHistory, and scripts/capture-history-goldens.mjs, which
// takes it from KataGo's BoardHistory (scripts/native/history-dump.cpp).
import { createHash } from 'node:crypto';

/** The recorded fields, in the order they're digested. */
export const STATE_FIELDS = [
  'colors', 'koLoc', 'encorePhase', 'isGameFinished', 'isPastNormalPhaseEnd', 'isScored', 'isNoResult', 'winner',
  'finalWhiteMinusBlackScore', 'whiteBonusScore', 'whiteHandicapBonusScore', 'consecutiveEndingPasses',
  'numApproxValidTurnsThisPhase', 'koRecapBlocked', 'superKoBanned', 'secondEncoreStartColors', 'nextPla',
  'passWouldEndPhase', 'passWouldEndGame', 'isFinalPhase', 'legal', 'passForKo', 'scoreNow', 'areaNow', 'getAreaNow',
  'independentLife',
];

/** The first 16 hex digits of the SHA-256 of the state's fields, in STATE_FIELDS order. */
export function digestState(state) {
  const text = JSON.stringify(STATE_FIELDS.map((k) => state[k]));
  return createHash('sha256').update(text).digest('hex').slice(0, 16);
}

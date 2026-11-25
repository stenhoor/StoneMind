/** Go rule sets, as KataGo defines them (`cpp/game/rules.cpp`, `Rules::parseRulesHelper`, KataGo 1.18). */

export type KoRule = 'SIMPLE' | 'POSITIONAL' | 'SITUATIONAL';
export type ScoringRule = 'AREA' | 'TERRITORY';
export type TaxRule = 'NONE' | 'SEKI' | 'ALL';
export type WhiteHandicapBonusRule = '0' | 'N' | 'N-1';

export interface Rules {
  koRule: KoRule;
  scoringRule: ScoringRule;
  taxRule: TaxRule;
  multiStoneSuicideLegal: boolean;
  hasButton: boolean;
  whiteHandicapBonusRule: WhiteHandicapBonusRule;
  friendlyPassOk: boolean;
}

const r = (
  koRule: KoRule,
  scoringRule: ScoringRule,
  taxRule: TaxRule,
  multiStoneSuicideLegal: boolean,
  whiteHandicapBonusRule: WhiteHandicapBonusRule,
  friendlyPassOk: boolean,
  hasButton = false,
): Rules => ({ koRule, scoringRule, taxRule, multiStoneSuicideLegal, hasButton, whiteHandicapBonusRule, friendlyPassOk });

const NAMED: Record<string, Rules> = {
  japanese: r('SIMPLE', 'TERRITORY', 'SEKI', false, '0', false),
  korean: r('SIMPLE', 'TERRITORY', 'SEKI', false, '0', false),
  chinese: r('SIMPLE', 'AREA', 'NONE', false, 'N', true),
  'chinese-ogs': r('POSITIONAL', 'AREA', 'NONE', false, 'N', true),
  'chinese-kgs': r('POSITIONAL', 'AREA', 'NONE', false, 'N', true),
  'stone-scoring': r('SIMPLE', 'AREA', 'ALL', false, '0', true),
  'ancient-territory': r('SIMPLE', 'TERRITORY', 'ALL', false, '0', false),
  aga: r('SITUATIONAL', 'AREA', 'NONE', false, 'N-1', true),
  bga: r('SITUATIONAL', 'AREA', 'NONE', false, 'N-1', true),
  'new-zealand': r('SITUATIONAL', 'AREA', 'NONE', true, '0', true),
  'tromp-taylor': r('POSITIONAL', 'AREA', 'NONE', true, '0', false),
};

/**
 * A named rule set, as the Analysis Engine's `rules` field accepts it.
 * @throws Error for names this engine doesn't support (rule sets with a button are refused for now).
 */
export function parseRules(name: string): Rules {
  const key = name.toLowerCase().replace(/[ _]/g, '-');
  const alias: Record<string, string> = { nz: 'new-zealand', newzealand: 'new-zealand', tromptaylor: 'tromp-taylor', french: 'aga' };
  const rules = NAMED[alias[key] ?? key];
  if (!rules) throw new Error(`unsupported rules "${name}"`);
  return { ...rules };
}

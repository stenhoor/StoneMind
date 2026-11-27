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

/**
 * Rules as KataGo's `Rules` holds them, with the komi (a C++ float). The ko rule may be `SPIGHT`, which KataGo parses but
 * no network from version 7 on supports (`supportedRules` changes it).
 */
export interface KataGoRules extends Omit<Rules, 'koRule'> {
  koRule: KoRule | 'SPIGHT';
  komi: number;
}

/** KataGo `Rules::MIN_USER_KOMI` and `MAX_USER_KOMI`. */
export const MIN_USER_KOMI = -400;
export const MAX_USER_KOMI = 400;

/** KataGo `Rules::komiIsIntOrHalfInt`. */
export function komiIsIntOrHalfInt(komi: number): boolean {
  return Number.isFinite(komi) && Number.isInteger(komi * 2) && Math.abs(komi * 2) < 2 ** 31;
}

const KO_RULES = ['SIMPLE', 'POSITIONAL', 'SITUATIONAL', 'SPIGHT'] as const;
const SCORING_RULES = ['AREA', 'TERRITORY'] as const;
const TAX_RULES = ['NONE', 'SEKI', 'ALL'] as const;
const WHB_RULES = ['0', 'N', 'N-1'] as const;

/** KataGo's rules parse error (an `IOError`). */
export class RulesParseError extends Error {}

const pick = <T extends string>(values: readonly T[], s: string, what: string): T => {
  if (!(values as readonly string[]).includes(s)) throw new RulesParseError(`Rules::parse${what}`);
  return s as T;
};

/**
 * KataGo `Rules::parseWhiteHandicapBonusRule`.
 * @throws RulesParseError, with KataGo's message.
 */
export function parseWhiteHandicapBonusRule(s: string): WhiteHandicapBonusRule {
  if ((WHB_RULES as readonly string[]).includes(s)) return s as WhiteHandicapBonusRule;
  throw new RulesParseError(`Rules::parseWhiteHandicapBonusRule: Invalid whiteHandicapBonus rule: ${s}`);
}

// KataGo Global::trim, toLower and toUpper work on ASCII only.
const trimAscii = (s: string): string => s.replace(/^[ \t\n\v\f\r]+|[ \t\n\v\f\r]+$/g, '');
export const toLowerAscii = (s: string): string => s.replace(/[A-Z]/g, (c) => c.toLowerCase());

const kg = (
  koRule: KataGoRules['koRule'],
  scoringRule: ScoringRule,
  taxRule: TaxRule,
  multiStoneSuicideLegal: boolean,
  hasButton: boolean,
  whiteHandicapBonusRule: WhiteHandicapBonusRule,
  friendlyPassOk: boolean,
  komi: number,
): KataGoRules => ({ koRule, scoringRule, taxRule, multiStoneSuicideLegal, hasButton, whiteHandicapBonusRule, friendlyPassOk, komi });

/** KataGo `Rules::getTrompTaylorish`. */
const trompTaylorish = (): KataGoRules => kg('POSITIONAL', 'AREA', 'NONE', true, false, '0', false, 7.5);

/** A rules object's value, read as nlohmann's `get<T>` reads it. */
export interface RulesObjectReader {
  /** The object's keys, in nlohmann's (sorted) order. */
  keys: readonly string[];
  getString(key: string): string;
  getBool(key: string): boolean;
  getFloat(key: string): number;
}

/**
 * KataGo `Rules::parseRules` (`parseRulesHelper` with komi allowed): a rule set's name, a JSON object (pass `object` to
 * read its fields; `s` is its text, for messages), or KataGo's compact form (`koSIMPLEscoreAREA...`).
 * @throws RulesParseError for anything KataGo refuses; the analysis engine reports it as `Could not parse rules: ${s}`.
 */
export function parseKataGoRules(s: string, object?: RulesObjectReader): KataGoRules {
  const lowercased = trimAscii(toLowerAscii(s));
  const named = (...names: string[]) => names.includes(lowercased);
  if (named('japanese', 'korean')) return kg('SIMPLE', 'TERRITORY', 'SEKI', false, false, '0', false, 6.5);
  if (named('chinese')) return kg('SIMPLE', 'AREA', 'NONE', false, false, 'N', true, 7.5);
  if (named('chineseogs', 'chinese_ogs', 'chinese-ogs', 'chinesekgs', 'chinese_kgs', 'chinese-kgs')) {
    return kg('POSITIONAL', 'AREA', 'NONE', false, false, 'N', true, 7.5);
  }
  if (named('ancientarea', 'ancient-area', 'ancient_area', 'ancient area', 'stonescoring', 'stone-scoring', 'stone_scoring', 'stone scoring')) {
    return kg('SIMPLE', 'AREA', 'ALL', false, false, '0', true, 7.5);
  }
  if (named('ancientterritory', 'ancient-territory', 'ancient_territory', 'ancient territory')) {
    return kg('SIMPLE', 'TERRITORY', 'ALL', false, false, '0', false, 6.5);
  }
  if (named('agabutton', 'aga-button', 'aga_button', 'aga button')) return kg('SITUATIONAL', 'AREA', 'NONE', false, true, 'N-1', true, 7.0);
  if (named('aga', 'bga', 'french')) return kg('SITUATIONAL', 'AREA', 'NONE', false, false, 'N-1', true, 7.5);
  if (named('nz', 'newzealand', 'new zealand', 'new-zealand', 'new_zealand')) return kg('SITUATIONAL', 'AREA', 'NONE', true, false, '0', true, 7.0);
  if (named('tromp-taylor', 'tromp_taylor', 'tromp taylor', 'tromptaylor')) return kg('POSITIONAL', 'AREA', 'NONE', true, false, '0', false, 7.5);
  if (named('goe', 'ing')) return kg('POSITIONAL', 'AREA', 'NONE', true, false, '0', true, 7.5);

  const rules = trompTaylorish();
  let komiSpecified = false;
  let taxSpecified = false;
  if (s.length > 0 && s[0] === '{') {
    if (!object) throw new RulesParseError('Could not parse rules');
    for (const key of object.keys) {
      if (key === 'ko') rules.koRule = pick(KO_RULES, object.getString(key), 'KoRule');
      else if (key === 'score' || key === 'scoring') rules.scoringRule = pick(SCORING_RULES, object.getString(key), 'ScoringRule');
      else if (key === 'tax') {
        rules.taxRule = pick(TAX_RULES, object.getString(key), 'TaxRule');
        taxSpecified = true;
      } else if (key === 'suicide') rules.multiStoneSuicideLegal = object.getBool(key);
      else if (key === 'hasButton') rules.hasButton = object.getBool(key);
      else if (key === 'whiteHandicapBonus') rules.whiteHandicapBonusRule = parseWhiteHandicapBonusRule(object.getString(key));
      else if (key === 'friendlyPassOk') rules.friendlyPassOk = object.getBool(key);
      else if (key === 'komi') {
        rules.komi = object.getFloat(key);
        komiSpecified = true;
        if (rules.komi < MIN_USER_KOMI || rules.komi > MAX_USER_KOMI || !komiIsIntOrHalfInt(rules.komi)) {
          throw new RulesParseError('Komi value is not a half-integer or is too extreme');
        }
      } else throw new RulesParseError(`Unknown rules option: ${key}`);
    }
  } else {
    // KataGo's legacy compact form.
    let rest = trimAscii(s);
    if (rest.length <= 0) throw new RulesParseError('Could not parse rules');
    const strip = (prefix: string): boolean => {
      const matches = rest.startsWith(prefix);
      if (matches) rest = rest.slice(prefix.length);
      rest = trimAscii(rest);
      return matches;
    };
    const fail = (): never => {
      throw new RulesParseError('Could not parse rules');
    };
    while (rest.length > 0) {
      if (strip('komi')) {
        let end = 0;
        while (end < rest.length && !/[A-Za-z \t\n\v\f\r]/.test(rest[end]!)) end++;
        // KataGo Global::tryStringToFloat: istream >> float, the whole string read.
        const text = rest.slice(0, end);
        if (!/^[+-]?(\d+\.?\d*|\.\d+)$/.test(text)) fail();
        const komi = Math.fround(Number(text));
        if (!Number.isFinite(komi) || komi > 1e5 || komi < -1e5) fail();
        rules.komi = komi;
        komiSpecified = true;
        rest = trimAscii(rest.slice(end));
        continue;
      }
      if (strip('ko')) {
        const k = KO_RULES.find((r) => strip(r));
        rules.koRule = k ?? fail();
        continue;
      }
      if (strip('scoring') || strip('score')) {
        const r = SCORING_RULES.find((x) => strip(x));
        rules.scoringRule = r ?? fail();
        continue;
      }
      if (strip('tax')) {
        const r = TAX_RULES.find((x) => strip(x));
        rules.taxRule = r ?? fail();
        taxSpecified = true;
        continue;
      }
      const flag = (prefix: string, set: (b: boolean) => void): boolean => {
        if (!strip(prefix)) return false;
        if (strip('1')) set(true);
        else if (strip('0')) set(false);
        else fail();
        return true;
      };
      if (flag('sui', (b) => (rules.multiStoneSuicideLegal = b))) continue;
      if (flag('button', (b) => (rules.hasButton = b))) continue;
      if (strip('whb')) {
        // N-1 before N, as KataGo checks.
        if (strip('0')) rules.whiteHandicapBonusRule = '0';
        else if (strip('N-1')) rules.whiteHandicapBonusRule = 'N-1';
        else if (strip('N')) rules.whiteHandicapBonusRule = 'N';
        else fail();
        continue;
      }
      if (flag('fpok', (b) => (rules.friendlyPassOk = b))) continue;
      fail();
    }
  }
  if (!taxSpecified) rules.taxRule = rules.scoringRule === 'TERRITORY' ? 'SEKI' : 'NONE';
  if (!komiSpecified) {
    // KataGo drops the default komi to 6.5 for territory scoring and 7 with a button.
    if (rules.scoringRule === 'TERRITORY') rules.komi = 6.5;
    else if (rules.hasButton) rules.komi = 7.0;
  }
  return rules;
}

/** KataGo's `operator<<` for `Rules`, as its warnings print rules (`koPOSITIONALscoreAREAtaxNONEsui1komi7.5`). */
export function rulesToString(r: KataGoRules): string {
  let out = `ko${r.koRule}score${r.scoringRule}tax${r.taxRule}sui${r.multiStoneSuicideLegal ? 1 : 0}`;
  if (r.hasButton) out += 'button1';
  if (r.whiteHandicapBonusRule !== '0') out += `whb${r.whiteHandicapBonusRule}`;
  if (r.friendlyPassOk) out += 'fpok1';
  // C++ prints a float with 6 significant digits.
  return out + `komi${Number(r.komi.toPrecision(6))}`;
}

/**
 * KataGo `ModelDesc::getSupportedRules` for network versions 7 and later (this engine loads 8 and later): the rules a
 * network supports nearest to `rules`, and whether they were supported as given.
 */
export function supportedRules(rules: KataGoRules): { rules: KataGoRules; supported: boolean } {
  const out = { ...rules };
  let supported = true;
  if (out.koRule === 'SPIGHT') {
    out.koRule = 'SITUATIONAL';
    supported = false;
  }
  if (out.hasButton && out.scoringRule !== 'AREA') {
    out.hasButton = false;
    supported = false;
  }
  return { rules: out, supported };
}

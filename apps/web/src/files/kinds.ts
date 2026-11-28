// The kinds of file the Files app takes, and how a file's kind is worked out from its name and its content.
//
// A kind is what a file is (a game, a problem, a KataGo network, a KataGo config), not where it is kept: folders are
// the player's own grouping, and a file of any kind can sit in any folder. A file whose name and content match none of
// the kinds is refused. Games and problems are both SGF; the content decides which (below), and the player can change
// it afterwards.
import { readWeightsHeader } from './weights.js';

export type FileKind = 'game' | 'problem' | 'weights' | 'config';

export const KINDS: readonly FileKind[] = ['game', 'problem', 'weights', 'config'];

/** The kind's name, as a sidebar entry (plural) and for one file (singular). */
export const KIND_NAMES: Readonly<Record<FileKind, { readonly plural: string; readonly one: string; readonly a: string }>> = {
  game: { plural: 'Games', one: 'Game', a: 'a game' },
  problem: { plural: 'Problems', one: 'Problem', a: 'a problem' },
  weights: { plural: 'Weights', one: 'KataGo network', a: 'a KataGo network' },
  config: { plural: 'Configs', one: 'KataGo config', a: 'a KataGo config' },
};

export const ACCEPTED = 'Files takes SGF games and problems, KataGo networks (.bin.gz, .txt.gz) and KataGo configs (.cfg).';

export type Evaluation = { readonly kind: FileKind; readonly why: string } | { readonly refused: string };

const SGF = /\.sgf$/i;
const WEIGHTS = /\.(bin|txt)\.gz$/i;
const CONFIG = /\.cfg$/i;

/** The first `bytes` of a file as text. */
async function head(file: Blob, bytes = 64 * 1024): Promise<string> {
  return new TextDecoder('utf-8', { fatal: false }).decode(await file.slice(0, bytes).arrayBuffer());
}

/**
 * Whether an SGF is a game or a problem, from its root properties: players (PB, PW) or a result (RE) make it a game;
 * otherwise setup stones (AB, AW) make it a problem; anything else is a game.
 */
export function sgfKind(text: string): { kind: 'game' | 'problem'; why: string } {
  const has = (prop: string) => new RegExp(`(^|[^A-Z])${prop}\\s*\\[`).test(text);
  const named = [['PB', 'Black player'], ['PW', 'White player'], ['RE', 'result']].filter(([p]) => has(p as string));
  if (named.length) return { kind: 'game', why: `It names the ${named.map(([p, what]) => `${what} (${p})`).join(', ')}.` };
  const setup = ['AB', 'AW'].filter(has);
  if (setup.length) return { kind: 'problem', why: `It sets up stones (${setup.join(', ')}) and names no players or result.` };
  return { kind: 'game', why: 'It names no players, and sets up no stones.' };
}

// Keys every KataGo config is built from; a config names at least one of them.
const CONFIG_KEYS = /^\s*(logDir|logFile|logSearchInfo|rules|koRule|scoringRule|taxRule|multiStoneSuicideLegal|hasButton|whiteHandicapBonus|maxVisits|maxPlayouts|maxTime|numSearchThreads|numAnalysisThreads|numSearchThreadsPerAnalysisThread|nnMaxBatchSize|nnCacheSizePowerOfTwo|nnMutexPoolSizePowerOfTwo|numNNServerThreadsPerModel|reportAnalysisWinratesAs|analysisPVLen|playoutDoublingAdvantage|ponderingEnabled|lagBuffer|resignThreshold|allowResignation|wideRootNoise|cpuctExploration|homeDataDir)\s*=/m;

/** Whether a .cfg file reads like a KataGo config: `key = value` lines and `#` comments, with at least one KataGo key. */
export function looksLikeKataGoConfig(text: string): boolean {
  const lines = text.split(/\r?\n/).map((l) => l.replace(/#.*$/, '').trim()).filter(Boolean);
  return lines.length > 0 && lines.every((l) => /^[A-Za-z0-9_.]+\s*=/.test(l)) && CONFIG_KEYS.test(text);
}

/** Works out a file's kind from its name and content, or says why Files won't take it. */
export async function evaluateKind(name: string, file: Blob): Promise<Evaluation> {
  if (SGF.test(name)) {
    const text = await head(file);
    if (!/^﻿?\s*\(\s*;/.test(text)) return { refused: `${name} is named .sgf but does not start like an SGF file.` };
    return sgfKind(text);
  }
  if (WEIGHTS.test(name)) {
    try {
      const h = await readWeightsHeader(file);
      return { kind: 'weights', why: `It starts like a KataGo network (${h.name}, model version ${h.modelVersion}).` };
    } catch {
      return { refused: `${name} is named like a KataGo network but does not start like one.` };
    }
  }
  if (CONFIG.test(name)) {
    // Read up to 1 MB; if the file is longer, leave out the last line, which may be cut.
    const LIMIT = 1024 * 1024;
    const text = await head(file, LIMIT);
    if (looksLikeKataGoConfig(file.size > LIMIT ? text.slice(0, text.lastIndexOf('\n')) : text)) return { kind: 'config', why: 'It holds KataGo settings.' };
    return { refused: `${name} is named .cfg but does not read like a KataGo config.` };
  }
  return { refused: `${name} is not a kind of file Files takes. ${ACCEPTED}` };
}

import { AnalysisError, type EngineDescriptor } from './engine.js';
import { JsonLineSession, type LineTransport } from './session.js';

/** The WebSocket subprotocol the local bridge speaks. */
export const BRIDGE_PROTOCOL = 'katago-analysis.v1';

/** The subprotocol entry that carries the pairing token. Subprotocols, unlike URLs, don't end up in history or logs. */
export const bridgeTokenProtocol = (token: string): string => `token.${token}`;

export interface BridgePairing {
  /** The bridge's WebSocket URL, e.g. "ws://127.0.0.1:6565". */
  url: string;
  /** The one-time pairing token the bridge printed. */
  token: string;
}

/**
 * Read a pairing from a URL fragment such as `#bridge=ws%3A%2F%2F127.0.0.1%3A6565&token=…`, as printed by the bridge.
 * @returns The pairing, or null when the fragment has none.
 */
export function parsePairingFragment(hash: string): BridgePairing | null {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const url = params.get('bridge');
  const token = params.get('token');
  if (!url || !token || !/^wss?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?\/?$/.test(url)) return null;
  if (!/^[A-Za-z0-9_-]{16,128}$/.test(token)) return null;
  return { url, token };
}

/** The descriptor of a native KataGo reached through the local bridge. */
export const LOCAL_KATAGO: EngineDescriptor = {
  id: 'katago-local',
  kind: 'local',
  displayName: 'KataGo on this computer',
};

/**
 * Connect to a native KataGo through the local bridge (`@stonemind/katago-bridge`).
 * @returns An engine that speaks KataGo's Analysis Engine protocol over the bridge's WebSocket.
 * @throws AnalysisError when the bridge can't be reached or refuses the pairing.
 */
export function connectBridge(pairing: BridgePairing, descriptor: EngineDescriptor = LOCAL_KATAGO): Promise<JsonLineSession> {
  return new Promise((resolve, reject) => {
    let socket: WebSocket;
    try {
      socket = new WebSocket(pairing.url, [BRIDGE_PROTOCOL, bridgeTokenProtocol(pairing.token)]);
    } catch (e) {
      reject(new AnalysisError(`can't open the bridge connection: ${(e as Error).message}`));
      return;
    }
    let opened = false;
    socket.addEventListener('error', () => {
      if (!opened) reject(new AnalysisError('the local bridge refused the connection or is not running'));
    });
    socket.addEventListener('open', () => {
      opened = true;
      if (socket.protocol !== BRIDGE_PROTOCOL) {
        socket.close();
        reject(new AnalysisError(`the bridge answered with an unexpected protocol "${socket.protocol}"`));
        return;
      }
      const transport: LineTransport = {
        send: (line) => socket.send(line),
        onLine: (receiver) => socket.addEventListener('message', (e) => receiver(String(e.data))),
        onClose: (receiver) =>
          socket.addEventListener('close', (e) => receiver(e.reason || `code ${e.code}`)),
        close: () => socket.close(1000, 'client closed'),
      };
      resolve(new JsonLineSession(descriptor, transport));
    });
  });
}

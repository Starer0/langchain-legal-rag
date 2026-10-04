export function createAuthClient({ fetchImpl = fetch, onSignedIn = () => {}, onSignedOut = () => {} } = {}) {
  let user = null;
  let csrf = null;
  let version = 0;
  let checking = null;
  const controllers = new Set();
  const responseControllers = new WeakMap();

  function clear() {
    const wasSignedIn = user !== null;
    user = null;
    csrf = null;
    version++;
    controllers.forEach(controller => controller.abort());
    controllers.clear();
    onSignedOut(wasSignedIn);
  }

  async function accept(data) {
    const changed = user?.id !== data.user.id;
    if (changed) {
      controllers.forEach(controller => controller.abort());
      controllers.clear();
      version++;
    }
    user = data.user;
    csrf = data.csrf_token;
    await onSignedIn(user, changed);
    return user;
  }

  async function errorFor(response, fallback) {
    let detail;
    try { detail = (await response.json()).detail; } catch { /* Use the safe fallback. */ }
    const requestId = response.headers.get('X-Request-ID');
    return new Error(`${typeof detail === 'string' ? detail : fallback}${requestId ? ` 请求编号：${requestId}` : ''}`);
  }

  return {
    get user() { return user; },
    get revision() { return version; },
    dispose() {
      user = null;
      csrf = null;
      version++;
      controllers.forEach(controller => controller.abort());
      controllers.clear();
    },
    async login(username, password) {
      const response = await fetchImpl('/api/auth/login', { method: 'POST', credentials: 'same-origin', cache: 'no-store',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
      if (!response.ok) throw await errorFor(response, '暂时无法登录，请稍后重试。');
      return accept(await response.json());
    },
    check() {
      if (checking) return checking;
      const started = version;
      checking = (async () => {
        const response = await fetchImpl('/api/auth/me', { credentials: 'same-origin', cache: 'no-store' });
        if (started !== version) return user;
        if (response.status === 401) { clear(); return null; }
        if (!response.ok) throw await errorFor(response, '暂时无法检查登录状态，请刷新重试。');
        const data = await response.json();
        if (started !== version) return user;
        return accept(data);
      })().finally(() => { checking = null; });
      return checking;
    },
    async request(url, options = {}) {
      const started = version;
      const controller = new AbortController();
      controllers.add(controller);
      const method = (options.method || 'GET').toUpperCase();
      const headers = { ...options.headers };
      if (!['GET', 'HEAD', 'OPTIONS'].includes(method) && csrf) headers['X-CSRF-Token'] = csrf;
      let response;
      try {
        response = await fetchImpl(url, { ...options, headers, credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
      } catch (error) {
        controllers.delete(controller);
        throw error;
      }
      if (started !== version) throw new Error('登录状态已变化，请重试。');
      if (response.status === 401) clear();
      // Stream bodies retain their controller until logout/account change. Trim
      // completed ordinary responses, which have no private stream to cancel.
      if (!response.headers.get('content-type')?.includes('text/event-stream')) controllers.delete(controller);
      else responseControllers.set(response, controller);
      return response;
    },
    release(response) {
      controllers.delete(responseControllers.get(response));
      responseControllers.delete(response);
    },
    async logout() {
      const response = await this.request('/api/auth/logout', { method: 'POST' });
      if (response.status === 401) return;
      if (!response.ok) throw await errorFor(response, '退出未完成，请重试。');
      clear();
    },
  };
}

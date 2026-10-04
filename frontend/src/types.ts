export type User = { id: string; username: string; level_id?: string };
export type Conversation = { id: string; title: string; updated_at?: string };
export type Source = {
  law_name?: string;
  title?: string;
  source?: string;
  article?: string;
  section?: string;
  pages?: number[];
  content?: string;
  [key: string]: unknown;
};
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  pending?: boolean;
};
export type Phase =
  | "checking"
  | "login"
  | "loading"
  | "ready"
  | "history-loading"
  | "history-error"
  | "load-error";
export interface ChatController {
  user: User | null;
  phase: Phase;
  conversations: Conversation[];
  activeId: string | null;
  messages: Message[];
  draft: string;
  title: string;
  status: string;
  error: string;
  busy: boolean;
  loginBusy: boolean;
  setDraft(value: string): void;
  login(username: string, password: string): Promise<void>;
  logout(): Promise<void>;
  check(): Promise<void>;
  retry(): Promise<void>;
  select(id: string): Promise<void>;
  create(): Promise<void>;
  rename(id: string, title: string): Promise<boolean>;
  remove(id: string): Promise<void>;
  send(): Promise<void>;
}

import { ChatView } from "./components";
import { useChat } from "./useChat";
export default function App() {
  const chat = useChat();
  return <ChatView key={chat.user?.id ?? "login"} chat={chat} />;
}

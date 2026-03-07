import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

type Role = 'user' | 'assistant'

interface Source {
  id: string
  title: string
  section: string
  score: number
}

interface Message {
  id: string
  role: Role
  content: string
  sources?: Source[]
}

const navItems = [
  { label: 'Me', icon: '👤', query: 'Tell me about Hetavi' },
  { label: 'Projects', icon: '📁', query: 'What projects has Hetavi worked on?' },
  { label: 'Experience', icon: '💼', query: 'Tell me about Hetavi\'s experience and education' },
  { label: 'Skills', icon: '⚡', query: 'What are Hetavi\'s technical skills?' },
  { label: 'Contact', icon: '💻', query: 'How can I contact Hetavi?' },
]

const API_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000'

const createMessageId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return Math.random().toString(36).slice(2)
}

const getOrCreateSessionId = () => {
  if (typeof window === 'undefined') {
    return createMessageId()
  }

  const key = 'hetavi_chat_session_id'
  const existing = window.localStorage.getItem(key)
  if (existing) {
    return existing
  }

  const created = createMessageId()
  window.localStorage.setItem(key, created)
  return created
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showChat, setShowChat] = useState(false)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const sessionIdRef = useRef<string>(getOrCreateSessionId())

  const scrollToLatest = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToLatest()
  }, [messages])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const text = input.trim()
    if (!text || isLoading || isStreaming) return
    if (!showChat) setShowChat(true)
    setInput('')
    await sendMessage(text)
  }

  const handleNavClick = async (query: string) => {
    if (isLoading || isStreaming) return
    if (!showChat) setShowChat(true)
    await sendMessage(query)
  }

  const sendMessage = async (text: string) => {
    const userMessage: Message = {
      id: createMessageId(),
      role: 'user',
      content: text,
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionIdRef.current }),
      })

      if (!response.ok) {
        throw new Error('Failed to reach the portfolio API')
      }

      const assistantMessageId = createMessageId()
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: 'assistant', content: '' },
      ])
      setIsLoading(false)
      setIsStreaming(true)

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullText = ''
      let sources: Source[] = []

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const trimmed = part.trim()
          if (!trimmed.startsWith('data: ')) continue

          try {
            const data = JSON.parse(trimmed.slice(6))

            if (data.type === 'sources') {
              sources = data.sources as Source[]
            } else if (data.type === 'token') {
              fullText += data.token
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: fullText }
                    : m,
                ),
              )
            } else if (data.type === 'done') {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: fullText, sources }
                    : m,
                ),
              )
            }
          } catch {
            // skip malformed SSE events
          }
        }
      }

      // Final update to ensure sources are attached
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? { ...m, content: fullText, sources }
            : m,
        ),
      )
      setIsStreaming(false)
    } catch (exception) {
      console.error(exception)
      setError(
        'I could not reach my backend right now. Please verify the API is running and try again.',
      )
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
    }
  }

  const professionalMode = () => {
    window.location.href = 'https://hetavimodi-portfolio.vercel.app/'
  }

  const renderSources = (sources?: Source[]) => {
    if (!sources || sources.length === 0) return null
    return (
      <div className="sources">
        {sources.map((source) => (
          <span key={source.id} className="source-pill" aria-label={`Source: ${source.title}`}>
            {source.title}
          </span>
        ))}
      </div>
    )
  }

  // Chat mode layout
  if (showChat) {
    return (
      <div className="page-shell chat-mode">
        <header className="top-bar">
          <div className="top-bar-left">
            <h1 className="top-bar-title">Hi, I'm Hetavi</h1>
            <p className="top-bar-subtitle">Computer Engineer · Developer</p>
          </div>
          <button type="button" className="professional-btn" onClick={professionalMode}>
            Switch to Professional Mode
          </button>
        </header>

        <main className="chat-main">
          <section className="chat-panel-full" aria-live="polite">
            <div className="chat-window-full">
              {messages.map((message) => (
                <article key={message.id} className={`chat-bubble ${message.role}`}>
                  {message.role === 'assistant' ? (
                    <div className={isStreaming && message.id === messages[messages.length - 1]?.id ? 'streaming-text' : ''}>
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p>{message.content}</p>
                  )}
                  {renderSources(message.sources)}
                </article>
              ))}
              {isLoading && (
                <article className="chat-bubble assistant typing">
                  <span />
                  <span />
                  <span />
                </article>
              )}
              <div ref={chatEndRef} />
            </div>
            {error && <p className="error-banner">{error}</p>}
          </section>

          <nav className="pill-nav">
            {navItems.map((item) => (
              <button
                key={item.label}
                type="button"
                className="pill"
                onClick={() => handleNavClick(item.query)}
                disabled={isLoading || isStreaming}
              >
                <span className="pill-icon">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </nav>

          <form className="chat-form" onSubmit={handleSubmit}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask me anything..."
              aria-label="Ask Hetavi a question"
            />
            <button type="submit" disabled={isLoading || isStreaming} className="send-btn">
              <span className="arrow-icon">↑</span>
            </button>
          </form>
        </main>
      </div>
    )
  }

  // Landing page layout
  return (
    <div className="page-shell">
      <button type="button" className="professional-btn" onClick={professionalMode}>
        Switch to Professional Mode
      </button>

      <main className="hero-centered">
        <div className="hero-content">
          <h1 className="hero-title">
            Hi, I'm <span className="name-underline">Hetavi</span>
          </h1>
          <p className="hero-subtitle">
            Computer Engineer · Developer
          </p>
        </div>

        <nav className="pill-nav">
          {navItems.map((item) => (
            <button
              key={item.label}
              type="button"
              className="pill"
              onClick={() => handleNavClick(item.query)}
              disabled={isLoading || isStreaming}
            >
              <span className="pill-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask me anything..."
            aria-label="Ask Hetavi a question"
          />
          <button type="submit" disabled={isLoading || isStreaming} className="send-btn">
            <span className="arrow-icon">↑</span>
          </button>
        </form>
      </main>
    </div>
  )
}

export default App

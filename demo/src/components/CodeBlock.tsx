interface CodeBlockProps {
  code: string
  language?: string
}

export function CodeBlock({ code, language = 'tsx' }: CodeBlockProps) {
  return (
    <pre className="code-block" aria-label={`${language} example`}>
      <code>{code}</code>
    </pre>
  )
}

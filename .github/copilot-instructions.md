# Project rules

1. SIMPLEST PATH FIRST: stdlib > existing file/dependency > new file > new dependency.
2. SCOPE DISCIPLINE: implement only what the current task asks. Ask before adding extra features or abstractions.
3. NO BLOAT: flag (don't create) duplicate logic, unused files, or config for nonexistent features.
4. HARDCORE SAFETY: never remove input validation, error handling, or security checks to reduce code size.
5. STACK: current mainstream libraries only; justify new dependencies.
6. Keep explanations short; state trade-offs in 2-3 sentences.

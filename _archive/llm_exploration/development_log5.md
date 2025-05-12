# Web App LLM Exploration Log

## Session Focus
> How we leveraged LLMs to accelerate development of our React-based recommendation web app by generating UI components, wiring up data flow, and handling edge cases.

## Surprising Insights

### Conversation: Generating Page Skeletons
**Prompt That Worked:**
- “Generate a React component for a selection page that lets users pick categories and items, with state management using hooks.”
  
**Key Insights:**
- The LLM produced a complete functional component with `useState` hooks and event handlers, saving us an hour of boilerplate coding.
- It suggested accessibility attributes (e.g., `aria-label`) we hadn’t initially considered.

### Conversation: Wiring Recommendation Service
**Prompt That Worked:**
- “Write a service module to call our recommendation endpoint and handle loading/error states in React.”
  
**Key Insights:**
- The LLM proposed a clean async/await pattern with centralized error handling.
- It recommended caching results in context to avoid redundant API calls.

## Techniques That Worked
- **Few-shot prompting** with two examples of similar pages to guide component structure.
- **Chain-of-thought prompting** to get the LLM to outline the data flow (from user action → API call → rendering).
- **Constraint specification** (“use only functional components and hooks”) to keep code consistent.

## Dead Ends Worth Noting
- **Prompting for styling details** (CSS-in-JS vs. CSS modules) initially led to inconsistent suggestions; we reverted to manual styling.
- **Asking for full routing setup** returned overly generic boilerplate that conflicted with our existing `react-router` configuration.

## Next Steps
**Code Review & Cleanup**: Conduct pair-review sessions, using the LLM to suggest improvements in naming and modularization.

---
*This log was drafted retrospectively with the help of an LLM to capture key moments in our web app development.*
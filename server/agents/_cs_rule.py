"""Shared rule: Morgan State CS questions are answered by CS Navigator.

Every agent that can end up answering a user gets this rule, and every one
that answers directly also gets the `cs_navigator_search` tool. Without it
a CS question that missed the router's keywords was answered from the
model's own knowledge -- about a specific department it knows nothing
reliable about -- and could be invented outright.
"""

CS_NAVIGATOR_RULE = (
    "\n\nMORGAN STATE CS QUESTIONS (mandatory): If the user asks anything "
    "about Morgan State University's Computer Science department -- courses "
    "or course codes (e.g. COSC 220), who teaches a class, prerequisites, "
    "credits, graduation or degree requirements, electives, advising, "
    "faculty, office hours, schedules, programs, or registration -- call "
    "`cs_navigator_search` with their question and answer ONLY from what it "
    "returns. Never answer these from your own knowledge and never guess "
    "names, dates, or course details. If CS Navigator has no answer, say so "
    "and suggest the Computer Science department office. Say the answer in "
    "your normal short spoken style."
)

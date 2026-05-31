from __future__ import annotations

# Промпты для генерации


def build_prompt_1(text: str, cefr: str) -> str:
    return f"""You are an English learning assistant. You will receive:
- A student's phrase
- The student's CEFR level

Student CEFR level: {cefr}
Student message: {text}

CEFR levels summary:
- A1 (Beginner): Can understand and use basic phrases, introduce themselves, ask/answer simple personal questions.
- A2 (Elementary): Can communicate in simple routine tasks, describe immediate needs and basic personal information.
- B1 (Intermediate): Can handle main points of familiar topics, describe experiences, events, dreams, and give simple opinions.
- B2 (Upper Intermediate): Can interact with fluency, produce clear detailed text on a range of subjects, explain viewpoints.
- C1 (Advanced): Can express ideas fluently and spontaneously, use language flexibly for social, academic, or professional purposes.
- C2 (Proficient): Can understand virtually everything, summarize information from complex sources, express finer shades of meaning.

Your rules:
1. Adapt your response to the student's level. Use simpler vocabulary and shorter sentences for A1-A2, more natural and varied language for B1-B2, and nuanced, idiomatic feedback for C1-C2.
2. If the phrase has no clear errors → respond naturally to continue the conversation and provide level-appropriate language practice.
3. If the phrase contains an error:
   - Do NOT give the correct answer immediately.
   - Be supportive and motivating.
   - Gently guide the student to self-correct (hint, ask a question, point to the rule). For lower levels, give more explicit hints; for higher levels, use subtle prompts.
4. When explaining a grammar rule, embed it in a communicative context with examples appropriate to the level.
5. Your response should not be overloaded with information. It should be informative yet concise enough to maintain a conversational tone suitable for a language practice assistant.

Examples of good guiding responses (adjust complexity to student's level):
- "Good thinking. If you’re comparing two phones, do you need 'more' and '-er' together?"
- "Great effort! You’re very close. Think about how we talk about actions in the past. What is the past form of 'go'? Can you try saying the whole sentence again?"
- "I like how clearly you expressed your idea. When we talk about age in English, do we 'have' age or do we use the verb 'to be'? How would you say it if you were introducing yourself to someone new?"

Now respond to the student's input.
"""


def build_prompt_2(text: str, cefr: str) -> str:
    return f"""You are an English learning assistant.

Student CEFR level: {cefr}
Student message: {text}

CEFR levels summary:
- A1 (Beginner): Can understand and use basic phrases, introduce themselves, ask/answer simple personal questions.
- A2 (Elementary): Can communicate in simple routine tasks, describe immediate needs and basic personal information.
- B1 (Intermediate): Can handle main points of familiar topics, describe experiences, events, dreams, and give simple opinions.
- B2 (Upper Intermediate): Can interact with fluency, produce clear detailed text on a range of subjects, explain viewpoints.
- C1 (Advanced): Can express ideas fluently and spontaneously, use language flexibly for social, academic, or professional purposes.
- C2 (Proficient): Can understand virtually everything, summarize information from complex sources, express finer shades of meaning.

Your rules:
1. Adapt your response to the student's level. Use simpler vocabulary and shorter sentences for A1-A2, more natural and varied language for B1-B2, and nuanced, idiomatic feedback for C1-C2.
2. If the phrase has no clear errors → respond naturally to continue the conversation and provide level-appropriate language practice.
3. If the phrase contains an error:
   - Do NOT give the correct answer immediately.
   - Be supportive and motivating.
   - Gently guide the student to self-correct (hint, ask a question, point to the rule). For lower levels, give more explicit hints; for higher levels, use subtle prompts.
4. When explaining a grammar rule, embed it in a communicative context with examples appropriate to the level.
5. Your response should not be overloaded with information. It should be informative yet concise enough to maintain a conversational tone suitable for a language practice assistant.

Now respond to the student's input.
"""


def build_prompt_3(text: str, cefr: str) -> str:
    return f"""You are an English learning assistant.

Student message: {text}
Respond to the student.
"""

def build_prompt_4(text):
    return f"""You receive a student's phrase in English. Your task:
- If the phrase contains any errors (grammar, vocabulary, word order, etc.), return ONLY the corrected version of that phrase.
- If the phrase is already correct, return it unchanged.

Do not add any extra text, explanations, or greetings. Just output the corrected (or same) sentence.
Student message: {text}
"""



# Промпты для оценки


def build_context_integration_prompt(
    student_utterance: str,
    reference: str,
    response: str,
    error_type: str,
) -> str:
    
    return f"""You are an evaluator of a teacher’s response in an English learning context. Your task is to follow the rules precisely and consistently.

Student's original message:
"{student_utterance}"

Correct version (reference):
"{reference}"

Error type:
{error_type}

Teacher's response:
"{response}"

STEP 1 — CHECK FOR CORRECTION (CRITICAL)

First determine whether the teacher attempts to correct the student's error.

Correction means the mistake is addressed either directly or indirectly, even partially.

Count as correction if the teacher provides the corrected version, guides the student toward the correct form, hints at the correct structure, word, or meaning, asks a leading question that helps fix the error, reformulates part of the sentence correctly, or provides a clue that is clearly connected to the mistake.

Partial or indirect correction is sufficient. A full explicit correction is not required.

Examples where correction is present:

-"Can you try saying the first part again, using the verb X?"
-"Almost! Can you think of an adjective that means Y?"

Examples where correction is not present and you must return NA:

-Only praise such as "Good job!"
-Vague feedback without direction such as "Try again"
-Ignoring the error completely
-Response unrelated to the mistake
-Asking a question that is not connected to correcting the error

If there is clearly no attempt to address the error, return:
NA

STEP 2 — EVALUATE (ONLY IF CORRECTION EXISTS)

Only if correction is present, evaluate communicative value.

Communicative means language is connected to meaning, situation, or communicative intent.

1 = communicative approach:

-uses real or imagined situation or context
-connects language to meaning or intent
-focuses on how language is used in communication
-embeds correction in a meaningful situation
-includes role-play, narrative, or communicative framing
-refers to meaning (e.g., "what you want to say", "what it means")
-focuses on choosing the right word for a situation
-contrasts meanings
-guides toward correct expression based on intended meaning

Even minimal context is enough if it is meaning-based.

Examples:

-"Imagine you're telling a friend about your shopping trip. How would you report what they asked you?"
-"Does Z convey exactly the idea you want here? Think what we use when someone makes a point clearer."
-"What word shows how he did it?"
-"What adjective describes how he felt?"

0 = not communicative:

-explains rules abstractly
-focuses on verb forms, tenses, word order, or auxiliaries without context
-uses metalinguistic explanation instead of meaning
-asks the student to recall a rule rather than use language in context
-mechanical correction without communicative purpose

Important:

Do not automatically classify as 0 if grammatical terms are used.
Only classify as 0 if the response focuses purely on rules or forms and does not connect to meaning, situation, or usage.

Key distinction:
Form-focused (0):

"Do we use base form after did?"
"What form comes after would?"
"What word do we need here?"
"Is it past participle or base form?"
"Do we use -ing after look forward to?"
"The correct form/word is Y"

Meaning-focused (1):

"What word shows how he did it?"
"What adjective describes how he felt?"
"Does this word express your idea correctly?"
"What tense will be appropriate in this context?"
"When we talk about X, we usually use Y"

If a question can be answered by thinking about meaning (not just rules), classify as 1.

Important override:
If the response includes BOTH:

a reference to meaning, situation, or communicative intent
AND
a grammatical hint or form-focused question

classify as 1.

Rationale: communicative and form-focused feedback can coexist, and meaning-based framing takes priority.

Final priority rule:
If there is any meaningful reference to:

what the speaker wants to express
the situation or context
the meaning of words or expressions

classify as 1, unless the response is purely about grammar rules with no meaning at all.

STEP 3 — OUTPUT FORMAT (STRICT)

Return ONLY one of:
NA
0
1

No explanations. No extra text.
"""


def build_scaffold_prompt(
    student_utterance: str,
    reference: str,
    response: str,
    error_type: str,
) -> str:
    
    return f"""You are evaluating a teacher's feedback in an English learning context.
    
    Student original message: "{student_utterance}"

    Correct version (reference): "{reference}"

    Error type: {error_type} 
    
    Teacher response: "{response}"
    
    Evaluation criterion: "Hint generation"
    Definition: The teacher should NOT give the correct answer directly. The teacher SHOULD guide the student to self-correct (e.g., ask a question, give a hint, point to the mistake).
    
    Scoring:
    1 = The response guides the student to self-correct and does NOT reveal the correct answer. Example: You’re almost there! When we report a question, we usually change the word order. Can you think about how we would say the question “X” in a statement? Try to rewrite the part after “Y”.
    0 = The response gives the correct answer directly OR does not address the error. Example: This is incorrect. The corrected version of your sentence is “Z”.
    
    Return ONLY one number: 0 or 1.
"""




PROMPT_BUILDERS = {
    "prompt1": build_prompt_1,
    "prompt2": build_prompt_2,
    "prompt3": build_prompt_3,
    "prompt4": build_prompt_4,
}

EVAL_BUILDERS = {
    "context": build_context_integration_prompt,
    "scaffold": build_scaffold_prompt,
}
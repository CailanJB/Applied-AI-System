# 🎵 Music Recommender Simulation Implemented with RAG

## DEMO LOOM WALKTHROUGH:
https://www.loom.com/share/6b5f931b22694bf3aff43562b6712160


## Project Summary
Project Objective: Expand Music recomender System by Incorporating a RAG pipeline
1. Scoring logic is still incorporated now we use RAG for retriveal and Generation


Modern recomender systems use a mix of signalks like behavioral signals like recording songs skipped, replayed, lked, or saved. They use item understanding like tempo, energy, speech, langugae, genre, to provide the best recomednations based on users preferences.

How the scoring system works during re-ranking phase :
Our content-based recomender will use a content-based filtering approach that uses song attributes to predict what users will love next. The flow is as follows: Our recomender compares each song to a user taste profile and scores how closely the song matches that profile using a mathematical formula with genre matchm mood match, acoustic preferences, and energy closeness so songs are rewarded for being near the users target values. The formula to calculate this is as follows: score = (genre match score + mood match + score + energy match score + acoustic prefernce score + dancebillity score). 
the scores for each category are calculated as follows: let similarity =  1 - (song category score - user profile category score) for each category. points = weight * (similarity)
Overview: compute per category points for each song related to user profile then add the points. if the scores between the song and user are an exact match provide +2.5 points if not caluclate using formula above. 

Our system priortizes the mood and energy first since in my opinion is what makes someone like specific songs. The final scores for all songs are ranked from worst to best and the top k songs are returned as recomendations. 

## How The System Works with RAG
PHASE 1: Convert local csv songs to text, to convert to embeddings to store in Vector DB.
PHASE 2: QUery process: validate user input gaurdraul/
Phase 3: Produce 384 float embedding of user query.
- two formats for user query
-plain text, in that acse we use SentenceTransformer Module to covert to vector embedding
- text + playlist url
- we assign a vector to the users playlist that captures the overall vibe of their playlist
phase 4: use embedding produced in phase 3 to perform semantic search aganist the vector index built in  return set of top 20 most similar songs.
phase 5: re-rank songs using existing coring logic taking into consideration the numerical data.
Phase 6: Generation, provide explanations for the recomended songs and provide spotify links for the user to see the song. 

## Overview on how the re-ranking system works(PHASE 4 OF RAG PIPELINE)
- From phase 4 via retrieval the top 20 songs are returned using vector similarity. This only uses categorical data so we pass these songs to our scoring system to also take into consideration the numerical data.
How the User-Preferences are Generated:
- In the front end user gets prompted what there prefernces are.
1. Start with user preferences
2. Read the users targets values(most prefered categories)(example prefernces values not every user will have)
    - genre: 0.444
    - mood: 0.23
    - energy: 0.40
    - acousticness: 0.24
    - dancebility: 0.30
3. Compare user profile to each song in dataset
4. score each song on a sacle from 0 to 10.
5. If user category and song category are an exact match award +3.0 points for that category, else 0 points for categorical data like mood and genre.
6. For numerical data(energy, acoustiness, danceability compute similarity score using provided formula)
  - Energy closeness: up to +2.5
  - Acousticness closeness: up to +1.5
  - Danceability closeness: up to +1.0
    - similarity = 1 - absolute difference
    - absolute different = |category in song weight - user perfernce weight | 
    - points = weight × similarity
  7. Add up all the points for each category and that is the score for the song
  8. sort the songs from highest to lowest score
  9. take top k songs and these are the recomended songs

Notes: System might prioritze energy closeness since for me energy is one of the biggets factors in liking a song. Therefore system may have bias to those who like energy in songs as the defying factor. 
Here is the algorithm recipe, end to end:



---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this

I learned that recomenders use a mix of two approach to make recomendations. Content based filtering and collaborative filtering. Content based filtering uses user preferences for recomendations where collaborative filtering makes recomendations on what other people like that you may also like. These recomenders all boil down to a mathmeatical formula that performs some distance calculation to see how similar your listenign history or watch history is to other people. However, bias can show up in these systems because not everyone likes the same things so the recomendations can be more in the direction of features that a large vareity of people perfer like mood or genre, where others may prfer lyricism or storytelling in music.

---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

1. Learning how RAG can be implemented into music recomendation systems was really interesting and amazing to see how it all boils down to mathematics. 

2. Using AI tools helped clarify implementation decisions like similarity calculations and score calculations. I needed to double check during the begging stages of the similarity because it was scaling to large numbers like 1000 which was too big to get good recomendations.

3. It suprised how recomendation systems can seem so complex but are really just simple algorithms that we learn during coursework just applied into a context.

4. I would try using a Machine learnign model using unsupervised learnign using methods like clustering to make recomendations. 

5. I think human judgement still matters when the AI is implementing core features. For example when I implemented my new scoring logic to take into account the RAG system AI suggested to just infer users preferences based on the songs in their playlist, or how they prompted the system. However, I came up with the more efficent idea to just ask the user before hand what are their prefernces in music to get better recomendations. In addition if they wanted more variety still giving them the option to change their prefernces. 
## Recomendation outputs in assests.md 


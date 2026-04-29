# 🎧 Model Card: Music Recommender Simulation with RAG

## 1. Model Name

ModelName "MoodChecked++" 


---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 
- Predicts what users will love next based on user profile preferences, what they input, and the overall vibe of their playlist.

Prompts:"How do popular stream platforms make recomendations" 

- What kind of recommendations does it generate: The system geenrates top k recomendations via RAG, using the csv dataset for the Retriveal to find similar songs. 
- What assumptions does it make about the user: It assumes user has prefered preferences with music. 
- Is this for real users or classroom exploration: Real users

---

## 3. How the Model Works  

Explain your scoring approach in simple language. For categorical data award points if exact match between user preference and song. for numerical calculate similarity with math formula. Add all points that is the score.  

Prompts:"Design a mathematical formula to score songs" 

- What features of each song are used (genre, energy, mood, etc): Mood, genre, energy,acousticness, dancebility are used for the main scoring logic after the songs have been retriebed during the RAG pipeline phase. 

- What user preferences are considered: Mood, genre, energy,acousticness, dancebility
- How does the model turn those into a score: For similarity checks for the RAG pipeline, the csv file rows are coverted into text that are turned into vector embeddings and stored in a vector DB for fast access. Then the user query are converted into a vector embeddign based on the text and during retrieval this query embedding gets macthced via a distance calculation to the songs in our vector DB. Then for the re-ranking, if a song's features have scores similar to user preference points are awarded.

- What changes did you make from the starter logic: Added score_song function to score the song that is used to compare to user preference. For query processing, first verifed the user's input is correct, ensuring safety and no malicious attacks.

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses. uses the song.csv file with categories such as mood, genre, energy, danceability. The csv file is the retrieval point for the RAG pipeline. 
- added new categories for better RAG embeddings. New categories for song objects: lyrics, description, relase year, popularity, and album song is in.

Prompts:"Add 80 songs to the csv file with diverse categories'"

- How many songs are in the catalog: 100
- What genres or moods are represented: mood,energy,tempo_bpm,valence,danceability,acousticness, description, release_year, popularity, lyrics.
- Did you add or remove data: added 80 new songs with new features. 
- Are there parts of musical taste missing in the dataset: Location of artist, some users prefer artist in their area. 

---

## 5. Strengths  

Where does your system seem to work well: Scoring logic works well by assigning points for exact matches for categorical data and uses similarity formula for numerical data. The RAG provides good distance calculations for users prefernces, specific songs they want and playlist features they have. 

Prompts:"List features for scoring logic algorithm for recomendations"

- User types for which it gives reasonable results: Users who prefer mood, genre,energy,dancebillity, and acousticness as top prefernces for their songs
- Any patterns you think your scoring captures correctly: The genre of the song 
- Cases where the recommendations matched your intuition: The mood of the songs

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 
Limitation in RAG system now: since the dataset used for retriveal is only a fixed amount there is high bias for these western american songs. In addition some of the songs in the dataset from the orignal repo are made up so spotify cant find any urls for them during the generation phase.
Prompts: Identiy potential biases in scoring logic
Limitations and Bias: Genre is very strong and can create genre bubbles that outweigh other categories. Dancebillity is somewhat flattened for everyone meaning two users with opposite dance preferences get similar recomendations. Energy gap can potentially bias toward mainstream artist when everyone may not be a fan of mainstream artist.
- Features it does not consider: Dancebility has little impact on recomendations 
- Genres or moods that are underrepresented: Dancebility 
- Cases where the system overfits to one preference: Mood based user preferences carry alot of influence 
- Ways the scoring might unintentionally favor some users: May favor users who perfer mood or genre. 
- In addition heavy bias towards only 100 songs and doesnt capture the entirety of spotifes catalog since spotify's catalog is so big it would be hard to read all this data on a single machine.
---

## 7. Evaluation  

How you checked whether the recommender behaved as expected: Ran pytest and functions in main. Asked copilot to genrate edge cases and evaluated that they still produced correct output. Genrated a front-end and human tested to see if the recomendations were accurate based on user prferences, user prompts, and a users playlist. 

Prompts: "Generate edge cases where categories have similar scores"

- Which user profiles you tested: Tested user profiles that prefered deep intense rock, chill lofi, and high energy pop, high energy, low dancebilitym r & b genre. 
- What you looked for in the recommendations: Based on profile top recomendation should have high mood score related to users perfered mood. 
- What surprised you: When commenting out mood category the recomendations mostly stayed the same.
- Any simple tests or comparisons you ran: Ran the test in the test folder. For each phase of the RAG pipeline implemented pytest for each function to ensure the correctness of the work. 

No need for numeric metrics unless you created some.

---

## 8. Future Work  

Ideas for how you would improve the model next: Instead of using a fixed amount for the RAG pipeline to use as retrieval, havem access to spotify's entire catalog by having cloud srvices to handle all these songs. In addition, have machine learning models to make accurate predictions. 

- 

Prompts:"Suggest some Machine Learnign models that are used for recomendation systems"

- Additional features or preferences: Maybe provide more prefernces like location of the artist. 
- Better ways to explain recommendations: Use more human language instead(eg people who listen to this artist also listen to these artist)
- Improving diversity among the top results: incorporate both collaborative filtering and content based filtering
- Handling more complex user tastes

---

## 9. Personal Reflection  

A few sentences about your experience: F

The bulk of the time for this project was coming up with a plan to implement a RAG pipeline effectively. Copilot offered some suggestions but some of the suggestions I felt were goign overboard and werent neccessary. For example when re-ranking the songs we also had to have the new user prefernces infered. However, copilot wanted to implement some algorithm to infer a users prefernces based on the prompt they inputted and their songs. However, instead of accpeting this I asked it to instead implement a feature that asked users what were there preferences and storing this in a user_prefernce object, allowing minimal changes to the code. 

Prompts: "How to update suer prefernecs to consider new RAG pipeline"
- What you learned about recommender systems: Most Services use a blend of content based and collaborative filtering.
- Something unexpected or interesting you discovered: Commenting out the mood category the results didnt change as much.
- How this changed the way you think about music recommendation apps: Cool to see how something we use everyday takes so much time to come up with and figure out.

Overall: This project was really informative and fun and gave me great insights on how AI is used as a collaborator and not a replacer. Having conversations with the AI telling it what is good to implement and what shouldn't be implemented always ensure the final result is stil in my hands, only the AI does the coding. However, the planning is still up to the engineer. 
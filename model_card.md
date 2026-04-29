# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name. "MoodChecked" 
Example: **VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 
- Predicts what users will love next based on user profile preferences

Prompts:"How do popular stream platforms make recomendations" 

- What kind of recommendations does it generate: the top k song recomendations  
- What assumptions does it make about the user: It assumes user has prefered preferences with music.
- Is this for real users or classroom exploration: Real users

---

## 3. How the Model Works  

Explain your scoring approach in simple language. For categorical data award points if exact match between user prefernce and song. for numerical calculate similarity with math formula. Add all points that is the score.  

Prompts:"Design a mathematical formula to score songs" 

- What features of each song are used (genre, energy, mood, etc.)Mood, genre, energy,acousticness, dancebility 
- What user preferences are considered: Mood, genre, energy,acousticness, dancebility
- How does the model turn those into a score: if a song's features have scores similar to user preference points are awarded.
- What changes did you make from the starter logic: Added score_song function to score the song that is used to compare to user preference.

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses. uses the song.csv file with categories such as mood, genre, energy, danceability. 

Prompts:"Add 5-10 songs to the csv file with diverse categories'"

- How many songs are in the catalog: 20 
- What genres or moods are represented: mood,energy,tempo_bpm,valence,danceability,acousticness
- Did you add or remove data: added 10 new songs with new features
- Are there parts of musical taste missing in the dataset: Location of artist, some users prfer artist in their area. 

---

## 5. Strengths  

Where does your system seem to work well: Scoring logic works well by assigning points for eaxct matches for categorical data and uses similarity formula for numerical data.

Prompts:"List features for scoring logic algorithm for recomendations"

- User types for which it gives reasonable results: Users who prefer mood and genre of songa
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

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected: Ran pytest and functions in main. Asked copilot to genrate edge cases and evaluated that they still produced correct output. 

Prompts: "Generate edge cases where categories have similar scores"

- Which user profiles you tested: Tested user profiles that prefered deep intesne rock, chill lofi, and high energy pop.
- What you looked for in the recommendations: Based on profile top recomendation should have high mood score related to users perfered mood. 
- What surprised you: When commenting out mood category the recomendations mostly stayed the same.
- Any simple tests or comparisons you ran: Ran the test in test_recommender.py

No need for numeric metrics unless you created some.

---

## 8. Future Work  

Ideas for how you would improve the model next: Implementing a machine learning model to predict recomendations for large amounts of users.

Prompts:"Suggest some Machine Learnign models that are used for recomendation systems"

- Additional features or preferences: Maybe provide more prefernces like location or artist.
- Better ways to explain recommendations: Use more human language instead(eg people who listen to this artist also listen to these artist)
- Improving diversity among the top results: incorporate both collaborative filtering and content based filtering
- Handling more complex user tastes

---

## 9. Personal Reflection  

A few sentences about your experience: Found it intersting learning how recomendation systems are made and how it all boils down to some mathematical formula.

Prompts: Prompts to fix scoring logic.

- What you learned about recommender systems: Most Services use a blend of content based and collaborative filtering.
- Something unexpected or interesting you discovered: Commenting out the mood category the results didnt change as much.
- How this changed the way you think about music recommendation apps: Cool to see how something we use everyday takes so much time to come up with and figure out.

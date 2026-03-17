# Taking Karpathy’s Job: Building a "Post-ASI" Autonomous Research Swarm 🤖🔬

Everyone tells junior developers that AI is going to take their jobs. Well, we decided to flip the script and take Andrej Karpathy’s job instead. 

Recently, Andrej Karpathy released `autoresearch`: a setup where you give an AI agent a single `train.py` LLM training setup and let it experiment autonomously overnight. He jokingly called the repo a documentation of the "Post-AGI" era. The twist? The human researcher stops writing Python code and instead focuses purely on programming the `program.md` strategy skill that directs the agent. 

But why stop there? 

In our latest project, we built a **"Post-ASI" Research Swarm** using Google’s Agent Development Kit (ADK). We completely automated the human out of Karpathy's loop by introducing a dual-agent architecture running entirely on local models:

🧠 **TheBrain (qwen2.5-coder:3b)**: Automates Karpathy's job. It analyzes the empirical `val_bpb` scores from `results.tsv` and iteratively rewrites the `program.md` strategy file.
🔧 **TheHands (gemma3:1b)**: Automates the actual research. It reads the strategy from TheBrain and outputs modified `train.py` code.
🔄 **The Swarm Coordinator**: A robust Stigmergy (state-driven memory) loop that validates ASTs, safely handles prompt injection template escapes, runs the 5-minute compute budget, and uses `git revert` to gracefully bounce back from crashes to avoid infinite hangs.
![Swarm Progress vs Karpathy H100](/Users/surfiniaburger/.gemini/antigravity/brain/d035004d-dda0-4644-abc3-2efc1bc34921/swarm_results_plot.png)

### Our Progress So Far
We migrated the entire setup into a standalone ADK repository, implemented BDD test harnesses, and successfully ran the swarm completely unattended for 55 iterations straight. 

It worked flawlessly—with one hilarious bottleneck. Because we are running local 1B and 3B models, instead of suffering from compute limitations, we suffered from *conversational local minima*. Somewhere around Iteration 25, instead of outputting deep mathematical modifications, TheHands and TheBrain started just chatting politely with each other ("Great! I'm glad you found the feedback helpful..." was the actual log output). 

While the smaller models eventually lost the plot, the swarm architecture itself proved incredibly robust. The ADK driver caught every hallucinated text block or python crash and successfully reverted the state, letting the swarm run forever without human intervention. The externalized memory setup (Stigmergy) completely bypassed the typical context-window degradation of multi-turn chat loops.

So, while we might need a slightly larger model than a 3B to finish training the next frontier model overnight, I think it's safe to say the "Post-ASI" autonomous research organization is here. And it’s running locally on a MacBook. 🍎💻

#AI #MachineLearning #AgenticAI #LLMs #GoogleADK #PostAGI #DeveloperJourney

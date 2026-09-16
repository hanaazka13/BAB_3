# Example 3.23 OpenAI Gym CartPole
import gymnasium as gym

# Membuat environment CartPole
env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(20):
    observation, info = env.reset()
    
    for t in range(100):
        print(observation)
        
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Episode selesai jika agen jatuh (terminated) atau batas waktu habis (truncated)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
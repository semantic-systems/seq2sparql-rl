# from collections import defaultdict
# next_time_steps_dict = defaultdict(list)
# num_episodes = 10
# for episode in range(num_episodes):
#     time_step = kgenv.reset()
#
#     next_time_steps_dict[0].append(time_step)
#
#     for rollout_step in range(1, num_rollout_steps+1):
#         # set current rollout step
#         # set_current_rollout_step(rollout_step)
#         for prev_time_step in next_time_steps_dict[rollout_step-1]:
#             action_step = policy.action(time_step)
#             next_time_step = kgenv.step(action_step)
#             episode_return += next_time_step.reward
#             traj = trajectory.from_transition(prev_time_step, action_step, next_time_step)
#             replay_buffer.add_batch(traj)
#
#             if not next_time_step.is_last():
#                 next_time_steps_dict[rollout_step].append(next_time_step)
#
#             #set rollout mode as True
#             select_actions = [] #"sample from distribution"
#
#             for select_action in select_actions:
#
#                 new_action = action_step.replace(select_action)
#                 new_next_time_step = kgenv.step(new_action)
#                 episode_return += new_next_time_step.reward
#                 traj = trajectory.from_transition(prev_time_step, new_action, new_next_time_step)
#                 # store additional experience
#                 replay_buffer.add_batch(traj)
#
#                 if not new_next_time_step.is_last():
#                     next_time_steps_dict[rollout_step].append(new_next_time_step)
#             # set rollout mode as False
#     #set_next_question_counter()
#
#
# # apply action function
# #episode_ended ?
#
# # environment step function
# reward = apply_action(action)
#
# if episode_ended:
#     return time_step_termination
# else:
#     if current_rollout_step >= num_rollout_steps:
#         return time_step_termination
#     return time_step_not_termination
#

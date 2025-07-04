So that we can rapidly test model changes on small hardware.

### Text

Sequential patterns:

- 1 2 3 4 -> 5
- a b c d -> e
- 1 1 2 3 5 -> 8
- `cat /usr/share/dict/words | sort -R | head -5 | sort` -> some subsequent alphabetically ordered word

Logic and math:

- eval 2 + 2 = -> 4 (randomly sample numbers and operators)
- eval 5 > 3 -> True
- count apple apple apple → 3

Language utils:

- reverse hello -> olleh
- count vowels hello -> 2
- first letter word -> w

### Vision
- how many dots in this image?
- what color is this image?
- what color is the center pixel?
- what color is the vertical line?

## Control/Robotics Tasks
- traverse this grid clockwise
- reach the goal (a*)
- grab object and release on target
  - state: `[agent_x, agent_y, obj_x, obj_y, target_x, target_y, holding_obj]`
  - state: `[...<mission>, ...<image>]`
  - actions: `[forward, turn_left, turn_right, grab, release`

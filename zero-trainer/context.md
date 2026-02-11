# 🧠 Проект: Tafl AlphaZero (микросеть)

## 🎯 Цель

Реализовать AlphaZero-подобную сеть для Tafl:

- вход: позиция 11×11  
- выход: policy (4840 ходов) + value  
- используется с MCTS (PUCT)  
- обучается на self-play  

---

# 📐 Архитектура сети

## 📥 Вход

Размер: **6 × 11 × 11**

Каналы:

1. Attackers (битовая карта)
2. Defenders
3. King
4. Side-to-move (плоскость из 0/1)
5. Corners (4 клетки)
6. Throne (центр)

Все каналы float32 (0.0 / 1.0).

---

## 🏗 Тело сети (trunk)

- Начальная свёртка:  
  `Conv2d(6 → 8, kernel=3, padding=1)`
- ReLU

- 3 residual блока

Каждый residual блок:

```
Conv2d(8 → 8, 3×3)
ReLU
Conv2d(8 → 8, 3×3)
Skip connection
ReLU
```

Всего:  
C = 8 каналов  
Depth = 3 residual blocks  

Это маленькая сеть для проверки гипотезы.

---

# 🎯 Policy head

Цель: 4840 ходов (121 клетки × 4 направления × 10 дистанций)

### Структура:

```
Conv1x1(8 → 40)
→ reshape в (40 × 11 × 11)
→ flatten в 4840
```

Почему 40?

Потому что:
- 4 направления × 10 дистанций = 40 типов движения  
- каждая клетка = origin  
- 40 × 121 = 4840  

Это spatial policy head, а не FC на 4840.

---

# 🎯 Value head

Структура:

```
Conv1x1(8 → 1)
→ flatten (121)
→ Linear(121 → 64)
→ ReLU
→ Linear(64 → 1)
→ tanh
```

Выход: scalar в диапазоне [-1, +1]

- +1 = победа текущей стороны  
- -1 = поражение  
- 0 = ничья / неопределённость  

---

# 📊 Loss функция

Общий loss:

```
Loss = PolicyLoss + ValueLoss
```

## Policy loss

Cross-entropy между:

- π_target (распределение из MCTS visits)
- softmax(logits) сети

Перед softmax:

- нелегальные ходы маскируются (logits = -inf)

## Value loss

MSE между:

- value_pred
- value_target (результат партии с точки зрения side-to-move)

## Код экспорта
```python
use crate::Board;
use crate::mcts::mcts::MCTSTree;
use crate::mcts::utils::move_to_policy_index;
use crate::position_export::BitPosition;
use crate::types::Side;
use std::io::{Write, Result};

pub const ACTIONS: usize = 121 * 4 * 10; // 4840
pub const LEGAL_MASK_BYTES: usize = (ACTIONS + 7) / 8; // 605

#[repr(C)]
#[derive(Clone)]
pub struct LegalMask {
    data: [u8; LEGAL_MASK_BYTES],
}

impl LegalMask {
    pub fn new() -> Self {
        Self {
            data: [0u8; LEGAL_MASK_BYTES],
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline]
    pub fn set(&mut self, action_index: usize) {
        debug_assert!(action_index < ACTIONS);

        let byte = action_index / 8;
        let bit = action_index % 8;

        self.data[byte] |= 1 << bit;
    }

    #[inline]
    pub fn is_set(&self, action_index: usize) -> bool {
        debug_assert!(action_index < ACTIONS);

        let byte = action_index / 8;
        let bit = action_index % 8;

        (self.data[byte] >> bit) & 1 == 1
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

fn compute_value(side_to_move: Side, result: Option<Side>) -> i8 {
    match result {
        None => 0, // draw / cutoff
        Some(winner) => {
            if winner == side_to_move {
                1
            } else {
                -1
            }
        }
    }
}



#[repr(C)]
#[derive(Clone, Copy)]
struct PolicyTarget {
    move_index: u16,
    visits: u16,
}

struct PendingSample {
    bit_position: BitPosition,
    legal_mask: LegalMask,
    policy: Vec<PolicyTarget>,
    value: i8,
}

impl PendingSample {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(self.bit_position.as_bytes())?;
        w.write_all(self.legal_mask.as_bytes())?;

        let policy_len = self.policy.len() as u16;
        w.write_all(&policy_len.to_le_bytes())?;

        for t in &self.policy {
            w.write_all(&t.move_index.to_le_bytes())?;
            w.write_all(&t.visits.to_le_bytes())?;
        }

        w.write_all(&[self.value as u8])?;
        Ok(())
    }

    pub fn set_value_from_result(&mut self, result: Option<Side>) {
        let stm_side = if self.bit_position.stm == 0 { Side::DEFENDERS } else { Side::ATTACKERS };
        self.value = compute_value(stm_side, result);
    }
}

impl MCTSTree {
    fn build_legal_mask_from_board(&mut self, board: &Board) -> LegalMask {
        let mut legal_mask = LegalMask::new();
        self.move_gen.generate_moves(board);

        for i in 0..self.move_gen.count {
            let mv = self.move_gen.moves[i];
            let move_index = move_to_policy_index(mv);
            legal_mask.set(move_index as usize);
        }

        legal_mask
    }

    /*
    Format:

    [BitPosition]          49 bytes
    [LegalMask]            605 bytes
    [policy_len]           u16
    [PolicyTarget × N]     N * 4 bytes
    [value]                i8
     */
    pub fn make_pending_sample(&mut self, board: &Board) -> PendingSample {
        let root = self.get_root();
        let mut policy: Vec<PolicyTarget> = vec![];


        for &child_id in *&root.children() {
            let node = self.get_node(child_id);
            let visits_f = node.visits();
            let visits_u16 = visits_f.round().min(u16::MAX as f32) as u16;

            if let Some(mv) = node.mv() {
                if visits_u16 == 0 {
                    continue; // skip moves that were not visited
                }

                let move_index = move_to_policy_index(mv);

                policy.push(PolicyTarget {
                    move_index,
                    visits: visits_u16,
                });
            }
        }

        PendingSample {
            bit_position:  BitPosition::from_board(board),
            legal_mask: self.build_legal_mask_from_board(board),
            policy,
            value: 0, // to be set later
        }
    }
}
```
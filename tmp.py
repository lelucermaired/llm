import json
with open('./checkpoints/qwen-gomoku-earlystop/ood_history.json', encoding='utf-8') as f:
    data = json.load(f)
print('best_step:', data['best_step'])
print('best_acc:', data['best_acc'])
for r in data['history']:
    print(f"step={r['step']} epoch={r['epoch']:.2f} ood={r['ood_acc']:.3f} loss={r['train_loss']:.4f}")
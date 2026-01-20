import numpy as np
from ..config import NUM_TASKS

class CLMetricsTracker:
    """Tracks CL metrics: Average Accuracy, Backward Transfer, Forgetting"""
    def __init__(self, num_tasks: int = NUM_TASKS):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.task_names = [f"Task {i+1}" for i in range(num_tasks)]

    def update(self, current_task: int, task_accuracies: dict):
        for task_id, acc in task_accuracies.items():
            self.accuracy_matrix[current_task][task_id] = acc

    def compute_average_accuracy(self, up_to_task: int):
        accs = []
        for task_id in range(up_to_task + 1):
            acc = self.accuracy_matrix[up_to_task][task_id]
            if acc > 0:
                accs.append(acc)
        return np.mean(accs) if accs else 0.0

    def compute_forgetting(self, up_to_task: int):
        if up_to_task == 0:
            return 0.0

        forgettings = []
        for task_id in range(up_to_task):
            max_acc = np.max(self.accuracy_matrix[:up_to_task+1, task_id])
            current_acc = self.accuracy_matrix[up_to_task][task_id]
            forgetting = max_acc - current_acc
            forgettings.append(forgetting)

        return np.mean(forgettings) if forgettings else 0.0

    def compute_backward_transfer(self):
        if self.num_tasks < 2:
            return 0.0

        bwt_scores = []
        for task_id in range(self.num_tasks - 1):
            initial_acc = self.accuracy_matrix[task_id][task_id]
            final_acc = self.accuracy_matrix[- 1][task_id]
            bwt = final_acc - initial_acc
            bwt_scores.append(bwt)

        return np.mean(bwt_scores) if bwt_scores else 0.0

    def get_summary(self):
        return {
            'Average Accuracy': self.compute_average_accuracy(self.num_tasks - 1),
            'Forgetting': self.compute_forgetting(self.num_tasks - 1),
            'Backward Transfer': self.compute_backward_transfer(),
            'Accuracy Matrix': self.accuracy_matrix.tolist()
        }

    def print_summary(self):
        print("\nCL METRICS SUMMARY")
        print("\nAccuracy Matrix (rows= after task X, cols= on task Y)")
        print("       ", end="")
        for i in range(self.num_tasks):
            print(f"T{i+1:2d}   ", end="")
        print()

        for i in range(self.num_tasks):
            print(f"After T{i+1}: ", end="")
            for j in range(self.num_tasks):
                acc = self.accuracy_matrix[i][j]
                if acc > 0:
                    print(f"{acc:.3f} ", end="")
                else:
                    print("  -   ", end="")
            print()

        print("Summary")
        print(f"  Average Accuracy:  {self.compute_average_accuracy(self.num_tasks-1):.4f}")
        print(f"  Forgetting:        {self.compute_forgetting(self.num_tasks-1):.4f}")
        print(f"  Backward Transfer: {self.compute_backward_transfer():.4f}")

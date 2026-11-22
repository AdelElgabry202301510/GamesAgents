import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, CheckButtons
import numpy as np
import threading
import time

class SudokuCSP:
    def __init__(self, initial_board=None, size=9):
        self.size = size
        self.subgrid_size = int(np.sqrt(size))
        self.variables = [(r, c) for r in range(size) for c in range(size)]
        self.initial_domains = {(r, c): list(range(1, size + 1))
                               for r in range(size) for c in range(size)}

        if initial_board is not None:
            for r in range(size):
                for c in range(size):
                    if initial_board[r][c] != 0:
                        self.initial_domains[(r, c)] = [int(initial_board[r][c])]

        self.reset()

    def reset(self):
        self.nodes = 0
        self.backtracks = 0
        self.solutions = 0

    def get_constrained_cells(self, var):
        r, c = var
        constrained = set()
        # Same row
        for col in range(self.size):
            if col != c:
                constrained.add((r, col))
        # Same column
        for row in range(self.size):
            if row != r:
                constrained.add((row, c))
        # Same subgrid
        start_r = (r // self.subgrid_size) * self.subgrid_size
        start_c = (c // self.subgrid_size) * self.subgrid_size
        for i in range(self.subgrid_size):
            for j in range(self.subgrid_size):
                cell = (start_r + i, start_c + j)
                if cell != var:
                    constrained.add(cell)
        return list(constrained)

    def is_consistent(self, var, value, assignment):
        r, c = var
        # Check row
        for col in range(self.size):
            if (r, col) in assignment and assignment[(r, col)] == value:
                return False
        # Check column
        for row in range(self.size):
            if (row, c) in assignment and assignment[(row, c)] == value:
                return False
        # Check subgrid
        start_r = (r // self.subgrid_size) * self.subgrid_size
        start_c = (c // self.subgrid_size) * self.subgrid_size
        for i in range(self.subgrid_size):
            for j in range(self.subgrid_size):
                cell = (start_r + i, start_c + j)
                if cell in assignment and assignment[cell] == value:
                    return False
        return True

    # MRV (Minimum Remaining Values)
    def select_unassigned_var(self, domains, assignment):
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: (len(domains[v]), -len(self.get_constrained_cells(v))))

    # LCV (Least Constraining Value)
    def order_domain_values(self, var, domains, assignment):
        def count_conflicts(value):
            conflicts = 0
            for neighbor in self.get_constrained_cells(var):
                if neighbor not in assignment:
                    if value in domains.get(neighbor, []):
                        conflicts += 1
            return conflicts

        values = list(domains[var])
        values.sort(key=count_conflicts)
        return values

    def forward_check(self, var, value, domains):
        new_domains = {v: list(domains[v]) for v in domains}
        removed = {v: [] for v in domains}
        new_domains[var] = [value]

        for neighbor in self.get_constrained_cells(var):
            if len(new_domains[neighbor]) > 1:  # Only process if not already assigned
                new_vals = []
                for val in new_domains[neighbor]:
                    if val == value:
                        removed[neighbor].append(val)
                    else:
                        new_vals.append(val)

                if not new_vals:
                    return None, removed

                new_domains[neighbor] = new_vals

        return new_domains, removed

    def backtrack_generator(self, domains=None, assignment=None):
        if domains is None:
            domains = {v: list(self.initial_domains[v]) for v in self.variables}
        if assignment is None:
            assignment = {}

        self.nodes += 1

        if len(assignment) == len(self.variables):
            self.solutions += 1
            yield ('solution', dict(assignment))
            return

        var = self.select_unassigned_var(domains, assignment)
        yield ('mrv', var, domains[var])

        lcv_order = self.order_domain_values(var, domains, assignment)
        yield ('lcv', var, lcv_order)

        for value in lcv_order:
            if not self.is_consistent(var, value, assignment):
                yield ('pruned', var, value)
                continue

            assignment[var] = value
            yield ('assign', var, value, dict(assignment))

            fc = self.forward_check(var, value, domains)
            if fc[0] is None:
                yield ('fc_fail', var, value, fc[1])
                del assignment[var]
                self.backtracks += 1
                yield ('unassign', var, dict(assignment))
                continue

            new_domains, removed = fc
            yield ('fc_ok', var, value, removed)

            for e in self.backtrack_generator(new_domains, assignment):
                yield e

            del assignment[var]
            self.backtracks += 1
            yield ('unassign', var, dict(assignment))

class SudokuDashboard:
    def __init__(self, board=None, speed=0.12):
        self.size = 9
        self.speed = speed

        if board is None:
            self.board = np.array([
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ])
        else:
            self.board = board

        self.csp = SudokuCSP(self.board, self.size)
        self.gen = None
        self.running = False
        self.assignment = {}
        self.domains = {v: list(self.csp.initial_domains[v]) for v in self.csp.variables}
        self.highlight = None
        self._build_ui()
        self.reset()

    def _build_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(12, 8))

        # board area
        self.ax_board = self.fig.add_axes([0.05, 0.12, 0.55, 0.82])
        self.ax_board.axis('off')
        self.ax_board.set_aspect('equal')

        # control buttons
        ax_step = self.fig.add_axes([0.63, 0.80, 0.12, 0.06])
        ax_run = self.fig.add_axes([0.76, 0.80, 0.12, 0.06])
        ax_reset = self.fig.add_axes([0.63, 0.72, 0.12, 0.06])
        ax_solve = self.fig.add_axes([0.76, 0.72, 0.12, 0.06])
        self.btn_step = Button(ax_step, 'Step')
        self.btn_run = Button(ax_run, 'Run')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_solve = Button(ax_solve, 'Solve All')
        self.btn_step.on_clicked(self.on_step)
        self.btn_run.on_clicked(self.on_run)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_solve.on_clicked(self.on_solve_all)

        # sliders
        ax_speed = self.fig.add_axes([0.63, 0.63, 0.25, 0.04])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.01, 1.0, valinit=self.speed)
        self.slider_speed.on_changed(self.on_speed)

        # metrics panel (text)
        self.ax_metrics = self.fig.add_axes([0.63, 0.12, 0.34, 0.38])
        self.ax_metrics.axis('off')
        self.metrics_text = self.ax_metrics.text(0, 1, '', va='top', family='monospace', fontsize=11)

        self.draw_board()
        plt.show(block=False)

    def reset(self):
        self.csp = SudokuCSP(self.board, self.size)
        self.csp.reset()
        self.gen = None
        self.running = False
        self.assignment = {}
        self.domains = {v: list(self.csp.initial_domains[v]) for v in self.csp.variables}
        self.highlight = None
        self.update_metrics()
        self.draw_board()

    def draw_board(self):
        size = self.size
        subgrid = self.csp.subgrid_size

        self.ax_board.clear()
        self.ax_board.set_xlim(0, size)
        self.ax_board.set_ylim(0, size)
        self.ax_board.set_xticks([])
        self.ax_board.set_yticks([])

        # Draw grid
        for r in range(size + 1):
            lw = 2 if r % subgrid == 0 else 0.5
            self.ax_board.axhline(y=r, color='black', linewidth=lw)
            self.ax_board.axvline(x=r, color='black', linewidth=lw)

        # Draw cells with domain sizes
        for (r, c) in self.csp.variables:
            # Background color
            color = '#ffffff'
            self.ax_board.add_patch(patches.Rectangle((c, size - r - 1), 1, 1,
                                                    facecolor=color, edgecolor='k', linewidth=0.3))

            # Domain size in small text
            ds = len(self.domains.get((r, c), []))
            if ds > 1:
                self.ax_board.text(c + 0.1, size - r - 0.9, f'{ds}',
                                 fontsize=6, color='blue', alpha=0.7)

        # Draw numbers (assignments and initial)
        for (r, c), value in np.ndenumerate(self.board):
            if value != 0:
                color = 'black'  # Initial numbers
                fontweight = 'bold'
                self.ax_board.text(c + 0.5, size - r - 0.5, str(int(value)),
                                 fontsize=16, ha='center', va='center',
                                 color=color, fontweight=fontweight)

        for (r, c), value in self.assignment.items():
            if self.board[r][c] == 0:  # Only draw if not initial
                self.ax_board.text(c + 0.5, size - r - 0.5, str(value),
                                 fontsize=16, ha='center', va='center',
                                 color='green', fontweight='bold')

        # Highlight current cell being tried
        if self.highlight:
            r, c, typ = self.highlight
            color = 'red' if typ == 'try' else 'green'
            self.ax_board.add_patch(patches.Rectangle((c, size - r - 1), 1, 1,
                                                    fill=False, edgecolor=color,
                                                    linewidth=3))

        self.fig.canvas.draw_idle()

    def update_metrics(self, extra=None):
        m = self.csp
        txt = [
            f"Sudoku {m.size}x{m.size}",
            f"Nodes visited: {m.nodes}",
            f"Backtracks: {m.backtracks}",
            f"Solutions: {m.solutions}",
            f"Assigned: {len(self.assignment)} / {len(m.variables)}",
            "",
            f"MRV chosen: {extra.get('mrv') if extra else ' - '}",
            f"LCV order: {extra.get('lcv') if extra else ' - '}",
            "",
            "Domain sizes:",
        ]

        # Show domain sizes in a grid
        for r in range(m.size):
            row_domains = []
            for c in range(m.size):
                ds = len(self.domains.get((r, c), []))
                row_domains.append(f"{ds:2d}")
            txt.append(" ".join(row_domains))

        self.metrics_text.set_text("\n".join(txt))
        self.fig.canvas.draw_idle()

    def handle_event(self, ev):
        typ = ev[0]
        if typ == 'mrv':
            _, var, domain = ev
            self.update_metrics(extra={'mrv': var, 'lcv': '-'}) # Removed 'n'

        elif typ == 'lcv':
            _, var, order = ev
            self.update_metrics(extra={'mrv': var, 'lcv': str(order[:5]) + '...' if len(order) > 5 else str(order)}) # Removed 'n'

        elif typ == 'assign':
            _, var, val, assign = ev
            self.assignment = dict(assign)
            # Update domains based on all assignments
            self.domains = {v: list(self.csp.initial_domains[v]) for v in self.csp.variables}
            for (r, c), value in self.assignment.items():
                fc, _ = self.csp.forward_check((r, c), value, self.domains)
                if fc:
                    self.domains = fc
            self.highlight = (var[0], var[1], 'try')
            self.draw_board()
            self.update_metrics()

        elif typ == 'fc_ok':
            _, var, val, removed = ev
            # Apply removals for display
            for cell, vals in removed.items():
                if cell in self.domains:
                    for vv in vals:
                        if vv in self.domains[cell]:
                            self.domains[cell].remove(vv)
            self.update_metrics()

        elif typ == 'fc_fail':
            _, var, val, removed = ev
            self.update_metrics()

        elif typ == 'pruned':
            _, var, val = ev
            self.update_metrics()

        elif typ == 'unassign':
            _, var, assign = ev
            self.assignment = dict(assign)
            # Rebuild domains
            self.domains = {v: list(self.csp.initial_domains[v]) for v in self.csp.variables}
            for (r, c), value in self.assignment.items():
                fc, _ = self.csp.forward_check((r, c), value, self.domains)
                if fc:
                    self.domains = fc
            self.highlight = None
            self.draw_board()
            self.update_metrics()

        elif typ == 'solution':
            _, assign = ev
            self.assignment = dict(assign)
            self.highlight = None
            self.draw_board()
            self.update_metrics()

    def on_step(self, event):
        if self.gen is None:
            self.gen = self.csp.backtrack_generator()
        try:
            ev = next(self.gen)
        except StopIteration:
            self.update_metrics()
            return
        self.handle_event(ev)

    def on_run(self, event):
        if self.gen is None:
            self.gen = self.csp.backtrack_generator()
        if self.running:
            self.running = False
            self.btn_run.label.set_text("Run")
        else:
            self.running = True
            self.btn_run.label.set_text("Pause")
            thread = threading.Thread(target=self._auto_loop, daemon=True)
            thread.start()

    def _auto_loop(self):
        while self.running:
            try:
                ev = next(self.gen)
            except StopIteration:
                self.running = False
                # schedule button label update on the GUI/main thread
                try:
                    self.fig.canvas.manager.window.after(0, lambda: self.btn_run.label.set_text("Run"))
                except Exception:
                    # fallback if window or manager not available
                    try:
                        self.btn_run.label.set_text("Run")
                    except Exception:
                        pass
                break

            # schedule handling the event on the GUI/main thread
            try:
                self.fig.canvas.manager.window.after(0, lambda ev=ev: self.handle_event(ev))
            except Exception:
                # fallback to direct call if scheduling unavailable
                self.handle_event(ev)

            time.sleep(getattr(self.slider_speed, 'val', self.speed))

    def on_reset(self, event):
        self.reset()

    def on_solve_all(self, event):
        if self.gen is None:
            self.gen = self.csp.backtrack_generator()
        self.running = True
        self.btn_run.label.set_text("Pause")
        def run_to_end():
            while True:
                try:
                    ev = next(self.gen)
                except StopIteration:
                    break
                try:
                    self.fig.canvas.manager.window.after(0, lambda ev=ev: self.handle_event(ev))
                except Exception:
                    self.handle_event(ev)

            self.running = False
            try:
                self.fig.canvas.manager.window.after(0, lambda: self.btn_run.label.set_text("Run"))
            except Exception:
                try:
                    self.btn_run.label.set_text("Run")
                except Exception:
                    pass
        thread = threading.Thread(target=run_to_end, daemon=True)
        thread.start()

    def on_speed(self, val):
        self.speed = val

if __name__ == "__main__":
    dash = SudokuDashboard(speed=0.12)
    plt.show()
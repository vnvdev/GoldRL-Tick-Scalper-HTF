"""
callbacks.py - Custom Stable-Baselines3 callbacks for monitoring live tick scalping progress.
"""

from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """
    Prints real-time worker progress, trade volume, local winrate (recent 200 trades),
    and cumulative winrate every `print_freq` steps.
    """

    def __init__(self, print_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq != 0:
            return True

        envs = self.training_env.envs if hasattr(self.training_env, "envs") else None
        rows = []
        for idx in range(self.training_env.num_envs):
            stats = None
            if envs is not None:
                base = envs[idx]
                while hasattr(base, "env"):
                    base = base.env
                stats = base.progress_stats
            else:
                stats = self.training_env.get_attr("progress_stats", indices=idx)[0]

            if stats is None:
                continue
            rows.append(
                f"  env{idx}: tick {stats['i']}/{stats['df_len']} ({stats['pct']:.1f}%) | so lenh: {stats['total']} "
                f"| winrate 200 lenh gan nhat: {stats['recent_wr']:.1f}% | winrate cong don: {stats['cum_wr']:.1f}% "
                f"| balance: {stats['balance']:.2f}"
            )
            if idx == 0:
                self.logger.record("custom/winrate_recent", stats["recent_wr"])
                self.logger.record("custom/winrate_cumulative", stats["cum_wr"])
                self.logger.record("custom/n_trades", stats["total"])
                self.logger.record("custom/tick_progress_pct", stats["pct"])

        print(f"[timestep {self.num_timesteps}]")
        print("\n".join(rows))
        return True

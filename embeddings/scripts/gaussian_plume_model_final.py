import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter


class WindField:
    def __init__(
        self,
        speed: float = 0.3,
        omega: float = 0.02,
        static_time: float = 10.0,
    ) -> None:
        self._speed = speed
        self._omega = omega
        self._static_time = static_time

    def __call__(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        omega = 0.02 if t > self._static_time else 0

        u = 2 * self._speed * np.sin(0.1 * np.pi * x - omega * t)
        v = self._speed * np.cos(0.1 * np.pi * y)
        return u, v


class AdvectionDiffusionEquation:
    D = 0.001

    def __init__(
        self,
        source_x: int,
        source_y: int,
        simulation_width: int,
        simulation_height: int,
        wind_field: WindField,
    ) -> None:
        self._nx = simulation_width
        self._ny = simulation_height

        self._dx = 1 / self._nx
        self._dy = 1 / self._ny

        x = np.linspace(0, 1, simulation_width)
        y = np.linspace(0, 1, simulation_height)
        self._X, self._Y = np.meshgrid(x, y)

        self._source = self._get_source(source_x, source_y)

        self._wind_field = wind_field

    def _get_source(self, x: int, y: int) -> np.ndarray:
        s = np.zeros((self._ny, self._nx))
        s[y, x] = 1
        return gaussian_filter(s, sigma=1)

    def __call__(self, t: float, u_flat: np.ndarray) -> np.ndarray:
        u = u_flat.reshape((self._nx, self._ny))

        # Compute second-order derivatives (Diffusion terms)
        du_dx2 = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / self._dx**2
        du_dy2 = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / self._dy**2

        # Compute first-order derivatives (Advection terms)
        du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * self._dx)
        du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * self._dy)

        v_x, v_y = self._wind_field(self._X, self._Y, t)

        # Note inverted sign for wind speed to simulate backwards in time computation
        du_dt = v_x * du_dx + v_y * du_dy + self.D * (du_dx2 + du_dy2) + self._source

        # Apply zero-gradient boundary conditions
        du_dt[0, :] = du_dt[-1, :] = 0
        du_dt[:, 0] = du_dt[:, -1] = 0

        return du_dt.flatten()


class GaussianPlumeModel:
    WIDTH = HEIGHT = 32

    UP_SAMPLING_FACTOR = 4

    SIMULATION_WIDTH = UP_SAMPLING_FACTOR * WIDTH
    SIMULATION_HEIGHT = UP_SAMPLING_FACTOR * HEIGHT

    STATIC_TIME = 5

    TIME_PER_MEASUREMENT = 0.5

    dt = 0.25

    def get_sensitivities_for_sensor(self, sensor_x: int, sensor_y: int, num_measurements: int) -> list[np.ndarray]:
        u_0_flat = np.zeros((self.SIMULATION_HEIGHT, self.SIMULATION_WIDTH)).flatten()

        pde = self._setup_advection_diffusion_equation(sensor_x=sensor_x, sensor_y=sensor_y)

        t_max = self.STATIC_TIME + self.TIME_PER_MEASUREMENT * num_measurements

        sol = solve_ivp(
            pde,
            [0, t_max],
            u_0_flat,
            method="RK45",
            t_eval=np.arange(0, t_max, self.dt),
        )

        u_sol = sol.y.reshape((self.SIMULATION_HEIGHT, self.SIMULATION_WIDTH, -1))

        down_sampled_array = self._down_sample_solution(u_sol)

        relevant_time_stamps = [
            int((self.STATIC_TIME + i * self.TIME_PER_MEASUREMENT) // self.dt) for i in range(num_measurements)
        ]

        return [down_sampled_array[:, :, i] for i in relevant_time_stamps]

    def _setup_advection_diffusion_equation(self, sensor_x: int, sensor_y: int) -> AdvectionDiffusionEquation:
        wind_field = WindField(static_time=self.STATIC_TIME)
        return AdvectionDiffusionEquation(
            source_x=self.UP_SAMPLING_FACTOR * sensor_x + self.UP_SAMPLING_FACTOR // 2,
            source_y=self.UP_SAMPLING_FACTOR * sensor_y + self.UP_SAMPLING_FACTOR // 2,
            simulation_height=self.SIMULATION_HEIGHT,
            simulation_width=self.SIMULATION_WIDTH,
            wind_field=wind_field,
        )

    def _down_sample_solution(self, sol: np.ndarray, down_sampling_with_mean: bool = False) -> np.ndarray:
        if down_sampling_with_mean:
            # Alternative with mean of cells
            reshaped_array = sol.reshape(
                self.HEIGHT,
                self.UP_SAMPLING_FACTOR,
                self.WIDTH,
                self.UP_SAMPLING_FACTOR,
                sol.shape[2],
            )
            return reshaped_array.mean(axis=(1, 3))
        return sol[
            self.UP_SAMPLING_FACTOR // 2 :: self.UP_SAMPLING_FACTOR,
            self.UP_SAMPLING_FACTOR // 2 :: self.UP_SAMPLING_FACTOR,
        ]


if __name__ == "__main__":
    model = GaussianPlumeModel()

    sensor_x = 25
    sensor_y = 25

    solutions = model.get_sensitivities_for_sensor(sensor_x=sensor_x, sensor_y=sensor_y, num_measurements=20)

    vmax = max([s.max() for s in solutions])

    for solution in solutions:
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(solution, cmap="viridis", vmin=0, vmax=vmax)
        plt.colorbar(cax, label="Sensitivity")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Gaussian Plume Model Footprint")
        plt.show()

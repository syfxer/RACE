from race.rewards import RARRReward, RARRRewardConfig


def test_race_reward_returns_expected_keys():
    rewarder = RARRReward(
        RARRRewardConfig(
            T=96,
            gamma=0.99,
            lambda_soc=100.0,
            soc_target=0.3,
            soc_deadband=0.02,
            soc_min=0.1,
            soc_max=0.9,
            delta_soc_ch_max=0.02,
            delta_soc_dis_max=0.02,
        )
    )
    info = rewarder.compute(t=0, soc_t=0.5, soc_next=0.48, voltage_cost_t=0.01)
    assert "reward" in info
    assert "r_voltage" in info
    assert "r_rarr" in info
    assert "recoverability_violation" in info

def main():
    import humanoid_kick.reward as reward
    import humanoid_kick.termination as termination
    import humanoid_kick.obs as obs
    print("Imports OK:", reward.__name__, termination.__name__, obs.__name__)

if __name__ == "__main__":
    main()

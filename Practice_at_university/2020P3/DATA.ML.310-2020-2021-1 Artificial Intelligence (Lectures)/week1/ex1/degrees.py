import csv
import sys
import numpy as np
from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    
    # TODO
    

    solution_found = False
    no_solution = False
    solution = list()
    solutions = []
    initial = Node(state=source, parent=None, action=None)
    frontier = QueueFrontier()
    frontier.add(initial)
    explored = set()
    i = 0
    while solution_found == False:

        if frontier.empty() == True:
            no_solution = True
            # solution_found = True
        
        node = frontier.remove()
        # print("\n\nNode in= ",node, ' i=', i)

        if node.state == target:
            # Return the solution
            solution_found = True
            # 生成解的办法
            solution = list()
            while node.parent is not None:
                
                pid, mid = node.state, node.action
                solution.append((mid, pid))
                node = node.parent
            solution.reverse()
            solutions.append(solution)
        explored.add(node)
        children = neighbors_for_person(node.state)
        for child in children:
            child_node = Node(state=child[1],parent=node, action=child[0])
            # 用边界树控制递进 呼应 node = frontier.remove()
            # 好好理解算法，理解了算法就可能完成而不必找解
            frontier.add(child_node)
            if child_node.state == target:
                # Return the solution
                solution_found = True
                solution = list()
                while child_node.parent is not None:
                    
                    pid, mid = child_node.state, child_node.action
                    solution.append((mid, pid))
                    child_node = child_node.parent
                solution.reverse()
                solutions.append(solution)
    l = []
    for i in range(len(solutions)):
        l.append(len(solutions[i]))
    j = np.argmin(l)   
    solution  = solutions[j]
    if solution_found == True:
        if no_solution == True:
            return None
        print(solution)
        return solution


def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


if __name__ == "__main__":
    main()
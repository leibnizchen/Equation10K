import random,re
import json
from itertools import product
from functools import lru_cache

OPS=("+","-","×","÷");PREC={"+":1,"-":1,"×":2,"÷":2}
INTEGERS=[str(i) for i in range(2,13)]
DECIMALS=["0.5","0.8","1.2","1.5","1.8","2.4","2.5","2.8","3.2","3.5","3.6","4.5","4.8","5.5","6.4","7.5",]
FRACTIONS=["1/2","1/3","2/3","1/4","3/4","1/5","2/5","3/5","4/5","1/6","5/6","1/8","3/8","5/8","7/8","2","3","5","7","8","9"]
QUESTION_TYPES=["addition","subtraction","multiplication","division","mixed_operations","parentheses","fraction","decimal","exponentiation"]

POWER_STYLE="random"  # "random"、"^"、"**"

def flat_templates(n,allowed_ops=OPS):return ["N"+"".join(op+"N" for op in ops) for ops in product(allowed_ops,repeat=n-1)]

@lru_cache(None)
def tree_shapes(l,r):
    if l==r:return (l,)
    out=[]
    for k in range(l,r):
        for a in tree_shapes(l,k):
            for b in tree_shapes(k+1,r):out.append((k,a,b))
    return tuple(out)

def render_tree(t,ops):
    if isinstance(t,int):return "N"
    k,a,b=t;return "("+render_tree(a,ops)+ops[k]+render_tree(b,ops)+")"

def bracket_templates(n,allowed_ops=OPS):
    out=set()
    for ops in product(allowed_ops,repeat=n-1):
        for t in tree_shapes(0,n-1):out.add(render_tree(t,ops)[1:-1])
    return list(out)

def get_ops(t):return [c for c in t if c in OPS]

def bracket_depth(t):
    d=m=0
    for c in t:
        if c=="(":d+=1;m=max(m,d)
        elif c==")":d-=1
    return m

def precedence_changes(ops):return sum(PREC[a]!=PREC[b] for a,b in zip(ops,ops[1:]))

def same_level_mix(ops):
    s=set(ops);return ("+" in s and "-" in s) or ("×" in s and "÷" in s)

def difficulty_score(t):
    ops=get_ops(t);n=t.count("N");u=len(set(ops));levels=len({PREC[x] for x in ops});depth=bracket_depth(t)
    score=(u>=2)+(u>=3)+(u==4)+2*same_level_mix(ops)+2*(levels==2)+precedence_changes(ops)+("-" in ops)+("÷" in ops)+(n==4)+2*(n>=5)+(3 if depth==1 else 6 if depth>=2 else 0)
    return 1 if score<=1 else 2 if score<=4 else 3 if score<=7 else 4 if score<=10 else 5

def build_template_library(min_numbers=3,max_numbers=5):
    lib={i:set() for i in range(1,6)}
    for n in range(min_numbers,max_numbers+1):
        for t in flat_templates(n)+bracket_templates(n):lib[difficulty_score(t)].add(t)
    return {k:tuple(v) for k,v in lib.items()}

TEMPLATE_LIBRARY=build_template_library()
ALL_TEMPLATES=tuple({t for v in TEMPLATE_LIBRARY.values() for t in v})

def filter_templates(qtype,difficulty):
    out=[]
    for t in TEMPLATE_LIBRARY[difficulty]:
        ops=set(get_ops(t));has_bracket="(" in t
        if qtype=="addition" and ops=={"+"}:out.append(t)
        elif qtype=="subtraction" and ops=={"-"}:out.append(t)
        elif qtype=="multiplication" and ops=={"×"}:out.append(t)
        elif qtype=="division" and ops=={"÷"}:out.append(t)
        elif qtype=="mixed_operations" and len(ops)>=2 and not has_bracket:out.append(t)
        elif qtype=="parentheses" and has_bracket:out.append(t)
        elif qtype in ("fraction","decimal"):out.append(t)
    return out

def power_pool(difficulty):
    out=[]
    for t in ALL_TEMPLATES:
        ops=get_ops(t);n=t.count("N");depth=bracket_depth(t);levels=len({PREC[x] for x in ops});u=len(set(ops))
        if difficulty==1 and depth==0 and n==3 and u==1 and ops[0] in ("+","×"):out.append(t)
        elif difficulty==2 and depth==0 and n<=4 and levels==1 and (u>=2 or ops[0] in ("-","÷")):out.append(t)
        elif difficulty==3 and depth==0 and levels==2:out.append(t)
        elif difficulty==4 and depth==1:out.append(t)
        elif difficulty==5 and depth>=2:out.append(t)
    return tuple(out)

POWER_POOLS={d:power_pool(d) for d in range(1,6)}

def integer_number():return random.choice(INTEGERS)
def decimal_number():return random.choice(DECIMALS)
def fraction_number():return random.choice(FRACTIONS)

def random_number():
    t=random.choices(("整数","小数","分数"),weights=(40,25,35),k=1)[0]
    return integer_number() if t=="整数" else decimal_number() if t=="小数" else fraction_number()

def mark_powers(t,difficulty):
    candidates=[i for i,c in enumerate(t) if c=="N" or c==")"];max_k=1 if difficulty<=2 else 2 if difficulty<=4 else 3
    chosen=set(random.sample(candidates,random.randint(1,min(max_k,len(candidates)))))
    return "".join(c+("@" if i in chosen else "") for i,c in enumerate(t))

def power_symbol():
    return random.choice(("^","**")) if POWER_STYLE=="random" else POWER_STYLE

def fill_template(t,number_func,ensure_fraction=False,exponents=(2,3)):
    count=t.count("N");nums=[number_func() for _ in range(count)]
    if ensure_fraction and not any("/" in x for x in nums):nums[random.randrange(count)]=fraction_number()
    sym=power_symbol();out=[];j=0
    for i,c in enumerate(t):
        if c=="N":
            x=nums[j];j+=1
            if i+1<len(t) and t[i+1]=="@" and "/" in x:x="("+x+")"
            out.append(x)
        elif c=="@":out.append(sym+str(random.choice(exponents)))
        else:out.append(c)
    return "".join(out)

def to_python_expression(expr):
    expr=re.sub(r'(?<![\d.])(\d+)/(\d+)(?![\d.])',r'(\1/\2)',expr)
    return expr.replace("×","*").replace("÷","/").replace("^","**")

def valid_expression(expr):
    try:
        v=eval(to_python_expression(expr),{"__builtins__":{}},{})
        return abs(v)<=100000
    except (ZeroDivisionError,ArithmeticError,SyntaxError,OverflowError):return False

def nearest_pool(qtype,difficulty):
    pool=filter_templates(qtype,difficulty)
    if pool:return pool
    for d in sorted(range(1,6),key=lambda x:abs(x-difficulty)):
        pool=filter_templates(qtype,d)
        if pool:return pool
    return []

def generate_question(question_type="random_combination",difficulty=3):
    difficulty=max(1,min(5,int(difficulty)))
    if question_type in ("随机","random_combination","以上方法random_combination"):
        pool=TEMPLATE_LIBRARY[difficulty]
        for _ in range(500):
            t=random.choice(pool)
            if random.random()<0.45:t=mark_powers(t,difficulty)
            e=fill_template(t,random_number,ensure_fraction=True)
            if valid_expression(e):return e
        raise RuntimeError("无法生成有效random_combination算式")
    if question_type in ("平方","立方","exponentiation","幂运算"):
        exponents=(2,) if question_type=="平方" else (3,) if question_type=="立方" else (2,3)
        pool=POWER_POOLS[difficulty]
        for _ in range(500):
            e=fill_template(mark_powers(random.choice(pool),difficulty),integer_number,exponents=exponents)
            if valid_expression(e):return e
        raise RuntimeError("无法生成有效exponentiation算式")
    pool=nearest_pool(question_type,difficulty)
    if not pool:raise ValueError(f"没有适用于{question_type}的模板")
    nf=fraction_number if question_type=="fraction" else decimal_number if question_type=="decimal" else integer_number
    for _ in range(500):
        e=fill_template(random.choice(pool),nf)
        if valid_expression(e):return e
    raise RuntimeError("无法生成有效算式")

def generate_questions(question_type="random_combination",difficulty=3,count=10):
    return [generate_question(question_type,difficulty) for _ in range(count)]



if __name__ == "__main__":

    for question_type in QUESTION_TYPES:
        quest_li = []
        idx = 0
        for i in range(200):
            for d in range(1, 6):
                questions = dict()
                questions["question_id"] = idx
                questions["question_type"] = question_type
                questions["difficulty"] = d
                q = generate_question(question_type,d)
                questions["question"] = q
                quest_li.append(questions)
                idx +=1
        with open(f"{"question_data"}/{question_type}.json", "w", encoding="utf-8") as f:
            json.dump(quest_li, f, ensure_ascii=False, indent=4)
    # quest_li = []
    # idx = 0
    # for i in range(200):
    #     for d in range(1, 6):
    #         questions = dict()
    #         questions["question_id"] = idx
    #         questions["question_type"] = "random_combination"
    #         questions["difficulty"] = d
    #         q = generate_question("random_combination", d)
    #         questions["question"] = q
    #         quest_li.append(questions)
    #         idx += 1
    # with open(f"{"question_data"}/{"random_combination"}.json", "w", encoding="utf-8") as f:
    #     json.dump(quest_li, f, ensure_ascii=False, indent=4)



"""Differential sanitizer oracle; seeds 1701 and 29102026 (5,000 trees each)."""
import importlib.util
from pathlib import Path
import random
import threading
from types import SimpleNamespace
import uuid

import pytest


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def pair():
    root=Path(__file__).resolve().parents[1]
    modules=[load(root/'src/common/source_identity.py','candidate'),load(Path(__file__).with_name('source_identity_oracle.py'),'oracle')]
    for module in modules:module.uuid=SimpleNamespace(uuid4=lambda:uuid.UUID(int=len(module._handles)+1))
    return modules


def register(modules,source):
    handle=uuid.UUID(int=len(modules[0]._handles)+1)
    for module in modules:module.uuid=SimpleNamespace(uuid4=lambda:handle)
    assert modules[0].public_source(source)==modules[1].public_source(source)


@pytest.mark.parametrize('seed',[1701,29102026])
def test_randomized_public_data_matches_verbatim_oracle(seed):
    new,old=pair();rng=random.Random(seed)
    sources=['rtsp://user:pass@host/feed?token=tok#frag','pipe ! password=abc','https://u:p@x/?api_key=雪','rtsp://a:b@h/']
    for source in sources:register((new,old),source)
    leaves=[None,True,False,0,1,1.5,float('nan'),float('inf'),float('-inf'),'RED','unknown','system','tok','tok_word','雪','xpassx','pass',sources[0],sources[0]+'suffix','token=leak','💡 café', '雪'*20000+' '+sources[2]]
    keys=['name','msg','reference','error','last_error','source','sources','source_sys','source_acq','PassWord','USER-ID','API_key','ordinary',None,42,('tuple',1)]
    def tree(depth=0):
        if depth>4 or rng.random()<.55:return rng.choice(leaves)
        if rng.random()<.35:return tuple(tree(depth+1) for _ in range(rng.randrange(4)))
        result={}
        for _ in range(rng.randrange(5)):
            key=rng.choice(keys)
            result[key]=rng.choice(sources+[None,42]) if key in ('source','source_sys','source_acq') else {'a':rng.choice(sources),'b':42} if key=='sources' else tree(depth+1)
        return result
    def normalize(value):
        # NaN identity/equality differs by container position; repr retains key/type distinctions.
        return repr(value)
    for _ in range(5000):
        value=tree();assert normalize(new.public_data(value))==normalize(old.public_data(value))
    deep='pass'
    for _ in range(35):deep={'name':deep,'list':(deep,) } if _<4 else [deep]
    assert normalize(new.public_data(deep))==normalize(old.public_data(deep))


def test_registry_changes_invalidate_cached_text_and_whole_view_discovery():
    modules=pair();new,old=modules
    for i in range(25):
        secret=f'secret{i}';value={'name':secret,'source':f'rtsp://user:{secret}@host/{i}'}
        before=new.safe_text(secret);assert before==old.safe_text(secret)
        register(modules,value['source'])
        assert new.safe_text(secret)==old.safe_text(secret)=='[redacted]'
        assert new.public_data(value)==old.public_data(value)
    value={'name':'new-secret','nested':{'source':'rtsp://u:new-secret@h/a'}}
    register(modules,value['nested']['source'])
    assert new.public_data(value)==old.public_data(value)
    assert new.safe_text('new-secret',fragments=False)=='new-secret'


def test_repeat_safe_text_uses_bounded_memoization():
    new,_=pair();calls=[];original=new._URL
    new._URL=SimpleNamespace(sub=lambda *a:calls.append(1) or original.sub(*a),finditer=original.finditer)
    for _ in range(100):assert new.safe_text('harmless free text')=='harmless free text'
    assert len(calls)==1
    for i in range(6000):new.safe_text(str(i))
    assert len(new._text_cache)<=4096


def test_registry_and_sanitizers_are_thread_safe():
    new,old=pair();errors=[];start=threading.Barrier(5)
    def reader():
        try:
            start.wait()
            for i in range(1000):new.public_data({'name':'concurrent-secret','items':['RED','token=bad',i]})
        except BaseException as error:errors.append(error)
    def writer():
        try:
            start.wait()
            for i in range(100):register((new,old),f'rtsp://u:concurrent-secret@h/{i}')
        except BaseException as error:errors.append(error)
    threads=[threading.Thread(target=reader) for _ in range(4)]+[threading.Thread(target=writer)]
    for thread in threads:thread.start()
    for thread in threads:thread.join(10)
    assert not any(t.is_alive() for t in threads);assert not errors
    assert new.public_data({'name':'concurrent-secret'})==old.public_data({'name':'concurrent-secret'})


def test_fragment_option_preserves_original_truthiness_for_unhashable_values():
    modules=pair();new,old=modules;register(modules,'rtsp://u:keyword-secret@h/feed')
    for fragments in ([],{},[1],{'yes':1},None,True,False,0,1):
        assert new.safe_text('keyword-secret',fragments=fragments)==old.safe_text('keyword-secret',fragments=fragments)
    mutable=[]
    for update in (False,True,False):
        mutable[:]=[1] if update else []
        assert new.safe_text('keyword-secret',fragments=mutable)==old.safe_text('keyword-secret',fragments=mutable)

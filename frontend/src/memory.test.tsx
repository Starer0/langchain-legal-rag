// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { MemorySettings } from './MemorySettings';
afterEach(()=>{cleanup();vi.useRealTimers();vi.restoreAllMocks();});
const initial = { core_text:'先给结论', extended_text:'Python初学者', enabled:true, revision:1, updated_at:null };

it('shows source quotes and inferred suggestion without claiming saved',async()=>{
  render(<MemorySettings owner='a' request={async()=>({...initial,facts:[{id:'f',layer:'core',content:'先给结论',basis:'declared',protected:false,
    sources:[{kind:'message',quote:'先说答案更容易懂',candidate_content:'先给结论',recorded_at:'2026-10-08T00:00:00Z'}]}],
    suggestions:[{id:'s',memory_revision:1,policy_epoch:1,proposal:{action:'remember',content:'近期在学习n8n',basis:'inferred',layer:'extended',relation:'new',sources:[]}}]})} onClose={vi.fn()}/>);
  await screen.findByText('近期在学习n8n');
  expect(screen.getByText('推测，待你确认')).toBeVisible();
  await userEvent.setup().click(screen.getByText('查看记忆来源'));
  expect(screen.getByText('先说答案更容易懂')).toBeVisible();
  expect(screen.queryByText('记忆已保存')).not.toBeInTheDocument();
});

it('old completed job cannot claim the current manually edited version was updated',async()=>{
  render(<MemorySettings owner='a' request={async()=>({...initial,revision:2,auto_accumulate:true,reflection_status:[{id:'j',status:'completed',result:{added:1,revision:1}}]})} onClose={vi.fn()}/>);
  await screen.findByLabelText('核心记忆');
  expect(screen.queryByText('已更新长期记忆')).not.toBeInTheDocument();
});

it('older failed jobs remain retryable after a later completion',async()=>{
  render(<MemorySettings owner='a' request={async()=>({...initial,auto_accumulate:true,reflection_status:[{id:'done',status:'completed',result:{added:0,revision:1}},{id:'failed',status:'failed'}]})} onClose={vi.fn()}/>);
  await screen.findByLabelText('核心记忆');expect(screen.getByRole('button',{name:'重试整理'})).toBeEnabled();
});

it('polling resumes after reload and never erases a draft made during the poll',async()=>{
  vi.spyOn(document,'visibilityState','get').mockReturnValue('visible');
  let poll!:()=>Promise<void>;
  const interval=window.setInterval.bind(window);
  vi.spyOn(window,'setInterval').mockImplementation((callback,delay,...args)=>{if(delay===5000){poll=callback as ()=>Promise<void>;return 123;}return interval(callback,delay,...args);});
  const request=vi.fn().mockResolvedValue({...initial,auto_accumulate:true,policy_epoch:1});
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()}/>);
  await screen.findByLabelText('核心记忆');
  const user=userEvent.setup();await user.click(screen.getByRole('button',{name:'重新加载'}));
  await waitFor(()=>expect(request).toHaveBeenCalledTimes(2));
  let resolve!:(value:unknown)=>void;
  request.mockReturnValueOnce(new Promise(r=>{resolve=r;}));
  let pending!:Promise<void>;await act(async()=>{pending=poll();});
  expect(request).toHaveBeenCalledTimes(3);
  fireEvent.change(screen.getByLabelText('核心记忆'),{target:{value:'尚未保存的编辑'}});
  await act(async()=>{resolve({...initial,core_text:'后台新增',revision:2,auto_accumulate:true,policy_epoch:1});await pending;});
  expect(screen.getByLabelText('核心记忆')).toHaveValue('尚未保存的编辑');
});

it('automatic accumulation is opt-in and does not claim a pending job is saved', async () => {
  const request=vi.fn(async(path:string,options?:RequestInit)=>path==='/api/memory/automation'
    ? {auto_accumulate:true,epoch:1} : {...initial,auto_accumulate:false,policy_epoch:0,reflection_status:[{id:'j',status:'pending'}]});
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()}/>);
  const user=userEvent.setup();const toggle=await screen.findByLabelText('自动积累长期记忆（后台处理）');
  expect(toggle).not.toBeChecked();await user.click(toggle);
  expect(request).toHaveBeenCalledWith('/api/memory/automation',expect.objectContaining({method:'PUT'}));
  expect(screen.queryByText('已更新长期记忆')).not.toBeInTheDocument();
  await waitFor(()=>expect(toggle).toBeEnabled());
  await user.click(toggle);
  expect(JSON.parse(request.mock.calls[2][1]!.body as string).expected_epoch).toBe(1);
});

it('conflicting memory displays old and proposed text for review', async () => {
  const request=vi.fn(async()=>({...initial,auto_accumulate:true,policy_epoch:1,suggestions:[{
    id:'s',memory_revision:1,policy_epoch:1,proposal:{text:'我喜欢简短回答',target:'我喜欢详细解释',relation:'conflict',layer:'core',source_quotes:['我喜欢简短回答']}}]}));
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()}/>);
  expect(await screen.findByText('我喜欢详细解释')).toBeVisible();
  expect(screen.getByText('我喜欢简短回答')).toBeVisible();
  expect(screen.getByRole('button',{name:'接受更新'})).toBeEnabled();
});

it('cannot confirm reload while a save is pending and unlocks after save', async () => {
  let finish:(value:unknown)=>void=()=>{};
  const pending=new Promise(resolve=>{finish=resolve;});
  const request=vi.fn(async(_path:string,options?:RequestInit)=>options?pending:initial);
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()}/>);
  const user=userEvent.setup();await user.type(await screen.findByLabelText('核心记忆'),'新');
  await user.click(screen.getByRole('button',{name:'重新加载'}));
  await user.click(screen.getByRole('button',{name:'保存记忆'}));
  expect(screen.getByRole('button',{name:'确认重新加载'})).toBeDisabled();
  await user.click(screen.getByRole('button',{name:'确认重新加载'}));
  finish({...initial,core_text:'先给结论新',revision:2});
  expect(await screen.findByText('记忆已保存')).toBeVisible();
  expect(screen.getByRole('button',{name:'关闭记忆设置'})).toBeEnabled();
  expect(request).toHaveBeenCalledTimes(2);
});

it('cannot overwrite memory when the initial read failed', async () => {
  const request=vi.fn(async()=>{throw Error('读取失败');});
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()}/>);
  expect(await screen.findByRole('alert')).toHaveTextContent('读取失败');
  expect(screen.getByLabelText('核心记忆')).toBeDisabled();
  expect(screen.getByRole('button',{name:'保存记忆'})).toBeDisabled();
});

it('saves both layers and disable with the current revision', async () => {
  const request = vi.fn(async (_path:string, options?:RequestInit) => options ? {...JSON.parse(options.body as string), revision:2} : initial);
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()} />);
  const user = userEvent.setup();
  const core = await screen.findByLabelText('核心记忆');
  await user.clear(core); await user.type(core, '保留来源');
  await user.click(screen.getByLabelText('在回答中使用记忆'));
  await user.click(screen.getByRole('button', {name:'保存记忆'}));
  expect(await screen.findByText('记忆已保存')).toBeVisible();
  expect(JSON.parse(request.mock.calls[1][1]!.body as string)).toEqual({ core_text:'保留来源', extended_text:'Python初学者', enabled:false, expected_revision:1 });
});

it('keeps draft when server reports conflict', async () => {
  const request = vi.fn(async (_path:string, options?:RequestInit) => {
    if (options) throw Error('记忆已在其他页面更新，请重新加载后编辑');
    return initial;
  });
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()} />);
  const user = userEvent.setup();
  const core = await screen.findByLabelText('核心记忆');
  await user.type(core, '，详细解释');
  await user.click(screen.getByRole('button', {name:'保存记忆'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('其他页面');
  expect(core).toHaveValue('先给结论，详细解释');
});

it('clear requires confirmation and preserves other layer', async () => {
  const request = vi.fn(async () => initial);
  render(<MemorySettings owner='a' request={request} onClose={vi.fn()} />);
  const user = userEvent.setup(); await screen.findByLabelText('核心记忆');
  await user.click(screen.getByRole('button', {name:'清空核心记忆'}));
  expect(screen.getByRole('button', {name:'取消'})).toHaveFocus();
  expect(screen.getByLabelText('核心记忆')).toHaveValue('先给结论');
  await user.click(screen.getByRole('button', {name:'确认清空'}));
  expect(screen.getByLabelText('核心记忆')).toHaveValue('');
  expect(screen.getByLabelText('扩展记忆')).toHaveValue('Python初学者');
  expect(request).toHaveBeenCalledTimes(1);
});

it('late response from previous owner cannot populate new owner', async () => {
  let resolve!: (value:typeof initial)=>void;
  const old = new Promise<typeof initial>(r => {resolve=r;});
  const request = vi.fn().mockReturnValueOnce(old).mockResolvedValue({...initial, core_text:'用户B'});
  const view=render(<MemorySettings owner='a' request={request} onClose={vi.fn()} />);
  view.rerender(<MemorySettings owner='b' request={request} onClose={vi.fn()} />);
  await waitFor(()=>expect(screen.getByLabelText('核心记忆')).toHaveValue('用户B'));
  resolve(initial);
  await waitFor(()=>expect(screen.getByLabelText('核心记忆')).toHaveValue('用户B'));
});

it('warns before closing unsaved changes and handles Escape', async () => {
  const close=vi.fn(), request=vi.fn(async()=>initial);
  render(<MemorySettings owner='a' request={request} onClose={close} />);
  const user=userEvent.setup(); await user.type(await screen.findByLabelText('核心记忆'),'新偏好');
  await user.keyboard('{Escape}');
  expect(close).not.toHaveBeenCalled();
  await user.click(screen.getByRole('button',{name:'放弃修改'}));
  expect(close).toHaveBeenCalledTimes(1);
});

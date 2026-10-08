import {test,expect} from '@playwright/test';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {resolve} from 'node:path';
import {randomUUID} from 'node:crypto';

for(const width of [1440,375]) test(`background memory survives browser close and server restart at ${width}px`,async({browser})=>{
  test.setTimeout(90000);
  const reservation=createServer();await new Promise<void>(r=>reservation.listen(0,'127.0.0.1',r));
  const address=reservation.address();if(!address||typeof address==='string')throw Error('No port');
  const port=address.port;await new Promise<void>(r=>reservation.close(()=>r()));
  const origin=`http://127.0.0.1:${port}`,root=resolve('test-results','reflection-'+randomUUID());
  let diagnostics='';
  const start=()=>{diagnostics='';const process=spawn('D:/conda/python.exe',['tests/fixtures/reflection_server.py',String(port),root],{cwd:'..',stdio:['ignore','pipe','pipe']});process.stderr.on('data',data=>{diagnostics+=data.toString();});return process;};
  let child=start();const context=await browser.newContext({viewport:{width,height:900}});
  const ready=()=>expect.poll(async()=>{if(child.exitCode!==null)throw Error(diagnostics);try{return (await fetch(origin+'/fixture')).status;}catch{return 0;}},{timeout:45000}).toBe(200);
  const exit=async()=>{const process=child;if(process.exitCode===null)await new Promise<void>(r=>{const timer=setTimeout(()=>{process.kill();r();},5000);process.once('exit',()=>{clearTimeout(timer);r();});});};
  try {
    await ready();let page=await context.newPage();await page.goto(origin);
    await page.getByLabel('用户名').fill('demo_ab');await page.getByLabel('密码').fill('MemoryTest123!');await page.getByRole('button',{name:'登录',exact:true}).click();
    await page.getByLabel('法律问题').fill('我喜欢简短回答');await page.getByRole('button',{name:'发送问题'}).click();
    await expect(page.locator('article.assistant').last()).toContainText('当前未开启自动积累，不会自动保存这条偏好');
    await fetch(origin+'/fixture/advance?seconds=120',{method:'POST'});
    expect((await (await fetch(origin+'/fixture')).json()).calls).toBe(0);
    if(width<600)await page.getByRole('button',{name:'打开对话列表'}).click();
    await page.getByRole('button',{name:'长期记忆',exact:true}).click();
    await page.getByLabel('自动积累长期记忆（后台处理）').click();
    await expect(page.getByLabel('自动积累长期记忆（后台处理）')).toBeChecked();
    await expect(page.getByRole('button',{name:'关闭记忆设置'})).toBeEnabled();await page.getByRole('button',{name:'关闭记忆设置'}).click();
    await page.getByLabel('法律问题').fill('我喜欢简短回答');await page.getByRole('button',{name:'发送问题'}).click();
    await expect(page.locator('article.assistant').last()).toContainText('收到你的回答风格偏好');
    await expect(page.locator('article.assistant').last()).toContainText('自动积累已开启');
    await expect(page.locator('article.assistant').last()).not.toContainText('参考来源');await page.close();
    await fetch(origin+'/fixture/stop?preserve=true',{method:'POST',signal:AbortSignal.timeout(5000)});await exit();child=start();await ready();
    await fetch(origin+'/fixture/advance?seconds=120',{method:'POST'});
    await expect.poll(async()=> (await (await fetch(origin+'/fixture')).json()).calls).toBe(1);
    page=await context.newPage();await page.goto(origin);
    if(width<600)await page.getByRole('button',{name:'打开对话列表'}).click();
    await page.getByRole('button',{name:'长期记忆',exact:true}).click();
    await expect(page.getByLabel('核心记忆')).toHaveValue('我喜欢简短回答');
    await expect(page.getByText('已更新长期记忆')).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  } finally {await context.close();await fetch(origin+'/fixture/stop',{method:'POST',signal:AbortSignal.timeout(5000)}).catch(()=>{});await exit();}
});

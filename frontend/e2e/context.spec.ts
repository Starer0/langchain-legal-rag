import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

for (const width of [1440,375]) test(`long context preserves early condition and isolates new chat at ${width}px`,async({browser})=>{
  test.setTimeout(60000);
  const reservation=createServer();
  await new Promise<void>(r=>reservation.listen(0,'127.0.0.1',r));
  const address=reservation.address(); if(!address || typeof address==='string')throw Error('No port');
  const port=address.port; await new Promise<void>(r=>reservation.close(()=>r()));
  const origin=`http://127.0.0.1:${port}`;
  const child=spawn('D:/conda/python.exe',['tests/fixtures/context_server.py',String(port),resolve('test-results','context-'+randomUUID())],{cwd:'..',stdio:'ignore'});
  const context=await browser.newContext({viewport:{width,height:900}});
  try {
    await expect.poll(async()=>{try{return (await fetch(origin+'/fixture')).status;}catch{return 0;}},{timeout:20000}).toBe(200);
    const page=await context.newPage();await page.goto(origin);
    await page.getByLabel('用户名').fill('demo_ab');await page.getByLabel('密码').fill('MemoryTest123!');
    await page.getByRole('button',{name:'登录',exact:true}).click();
    await page.getByLabel('法律问题').fill('那约定呢');await page.getByRole('button',{name:'发送问题'}).click();
    await expect(page.locator('article.assistant').last()).toContainText('保留条件：合同期限两年');
    await page.reload();await expect(page.locator('article.assistant').last()).toContainText('保留条件：合同期限两年');
    expect((await (await fetch(origin+'/fixture')).json()).summary_calls).toBe(1);
    if(width<600)await page.getByRole('button',{name:'打开对话列表'}).click();
    await page.getByRole('button',{name:/^新建对话/}).click();
    await page.getByLabel('法律问题').fill('工资是多少');await page.getByRole('button',{name:'发送问题'}).click();
    await expect(page.locator('article.assistant').last()).toContainText('保留条件：无早期条件');
    expect((await (await fetch(origin+'/fixture')).json()).summary_calls).toBe(1);
  } finally {
    await context.close();
    await fetch(origin+'/fixture/stop',{method:'POST'}).catch(()=>{});
    if(child.exitCode===null)await new Promise<void>(r=>{child.once('exit',()=>r());setTimeout(()=>{child.kill();r();},5000);});
  }
});

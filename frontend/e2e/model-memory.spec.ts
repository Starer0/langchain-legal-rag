import {test,expect} from '@playwright/test';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {resolve} from 'node:path';
import {randomUUID} from 'node:crypto';

for(const width of [1440,375]) test(`semantic mixed answer, receipt, sources and review at ${width}px`,async({browser})=>{
  test.setTimeout(90000);
  const reservation=createServer();await new Promise<void>(r=>reservation.listen(0,'127.0.0.1',r));
  const address=reservation.address();if(!address||typeof address==='string')throw Error('No port');
  const port=address.port;await new Promise<void>(r=>reservation.close(()=>r()));
  const origin=`http://127.0.0.1:${port}`,root=resolve('test-results','semantic-'+randomUUID());
  let diagnostics='';
  const child=spawn('D:/conda/python.exe',['tests/fixtures/reflection_server.py',String(port),root,'semantic'],{cwd:'..',env:{...process.env,PYTHONUTF8:'1'},stdio:['ignore','pipe','pipe']});
  child.stderr.on('data',data=>{diagnostics+=data.toString();});
  const context=await browser.newContext({viewport:{width,height:900}});
  try {
    await expect.poll(async()=>{if(child.exitCode!==null)throw Error(diagnostics);try{return (await fetch(origin+'/fixture')).status;}catch{return 0;}},{timeout:45000}).toBe(200);
    const page=await context.newPage();await page.goto(origin);
    await page.getByLabel('用户名').fill('demo_ab');await page.getByLabel('密码').fill('MemoryTest123!');await page.getByRole('button',{name:'登录',exact:true}).click();
    const openMemory=async()=>{if(width<600)await page.getByRole('button',{name:'打开对话列表'}).click();await page.getByRole('button',{name:'长期记忆',exact:true}).click();};
    const send=async(q:string)=>{const count=await page.locator('article.assistant').count();await page.getByLabel('法律问题').fill(q);await page.getByRole('button',{name:'发送问题'}).click();await expect(page.locator('article.assistant')).toHaveCount(count+1);await expect.poll(async()=> (await (await fetch(origin+'/fixture')).json()).running).toBe(0);};
    await send('请记住先给结论；试用期最长多久？');
    await expect(page.locator('article.assistant').last()).toContainText('没有足够依据');
    await expect(page.getByText('已保存长期记忆。', {exact:true})).toBeVisible();
    await page.reload();await expect(page.getByText('已保存长期记忆。',{exact:true})).toBeVisible();
    await openMemory();await expect(page.getByLabel('核心记忆')).toHaveValue('先给结论');
    await page.getByText('查看记忆来源',{exact:true}).click();await expect(page.getByRole('blockquote')).toContainText('请记住先给结论');
    await page.getByLabel('自动积累长期记忆（后台处理）').click();
    await expect(page.getByRole('button',{name:'关闭记忆设置'})).toBeEnabled();await page.getByRole('button',{name:'关闭记忆设置'}).click();
    await send('n8n的Webhook怎么接收POST？');
    await expect(page.locator('article.assistant').last()).toContainText('超出了本助手的法律服务范围');
    await expect(page.locator('article.assistant').last()).not.toContainText('直接给出');
    await send('以后涉及纠纷，请先提醒我保留证据，记住这个偏好。顺便推荐几款游戏。');
    await expect(page.locator('article.assistant').last()).toContainText('已保存长期记忆');
    await expect(page.locator('article.assistant').last()).toContainText('非法律业务请求超出了');
    await page.reload();
    await expect(page.locator('article.assistant').last()).toContainText('非法律业务请求超出了');
    await send('劳动仲裁申请需要什么材料？');await send('劳动仲裁提交申请后有哪些步骤？');
    await fetch(origin+'/fixture/advance?seconds=120',{method:'POST'});
    await openMemory();await expect(page.getByText('近期在学习劳动仲裁',{exact:true})).toBeVisible({timeout:15000});
    await expect(page.getByText('推测，待你确认')).toBeVisible();
    await page.getByRole('button',{name:'接受更新'}).click();await expect(page.getByLabel('扩展记忆')).toHaveValue('近期在学习劳动仲裁');
    await page.getByLabel('核心记忆').fill('尚未保存草稿');await page.waitForTimeout(5100);
    await expect(page.getByLabel('核心记忆')).toHaveValue('尚未保存草稿');
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  } finally {
    await context.close();await fetch(origin+'/fixture/stop',{method:'POST',signal:AbortSignal.timeout(5000)}).catch(()=>{});
    if(child.exitCode===null)await new Promise<void>(r=>{const timer=setTimeout(()=>{child.kill();r();},5000);child.once('exit',()=>{clearTimeout(timer);r();});});
  }
});

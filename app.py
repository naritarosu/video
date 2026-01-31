# app.py
import gradio as gr

import story_planner as sp
import generate


def main():
    print(generate.ensure_reforge_running(timeout_sec=180.0))

    with gr.Blocks(title="Story Planner + Generate (Image)") as app:
        gr.Markdown("# PV")

        with gr.Tabs():
            # -------------------------
            # Tab 1: Story Planner
            # -------------------------
            with gr.TabItem("1) Story Planner"):
                gr.Markdown("## Story Planner")

                product_in = gr.Textbox(label="商品名", placeholder="例：コーラ")
                theme_in = gr.Textbox(label="テーマ", placeholder="例：夏に飲みたくなる爽快感")
                appearance_prefix_in = gr.Textbox(
                    label="キャラクターの外見",
                    placeholder="例：single protagonist, consistent character design, same outfit, ...",
                )

                btn_make = gr.Button("Generate Story + Prompts", variant="primary")

                story_out = gr.Textbox(label="ストーリー", lines=10)
                p1_out = gr.Textbox(label="Scene1 Prompt (EN)", lines=4)
                p3_out = gr.Textbox(label="Scene3 Prompt (EN)", lines=4)
                p4_out = gr.Textbox(label="Scene4 Prompt (EN)", lines=4)

                prompts_state = gr.State({})

                prompt_pick = gr.Radio(
                    label="Generateタブへ送るプロンプトを選択",
                    choices=["Scene1", "Scene3", "Scene4"],
                    value="Scene1",
                    interactive=True,
                )
                btn_send = gr.Button("↓ 選択したPromptをGenerateタブの入力欄へ送る")

                def _make_story(product, theme, prefix):
                    if not product or not theme:
                        raise gr.Error("商品名とテーマは必須です。")
                    story, prompts = sp.plan_story_and_prompts(product, theme, prefix or "")

                    story_text_lines = []
                    for i in [1, 2, 3, 4]:
                        story_text_lines.append(f"[{i}] シーン: {story[i]['scene']}")
                        story_text_lines.append(f"    テロップ: {story[i]['telop']}")
                    story_text = "\n".join(story_text_lines)

                    return story_text, prompts[1], prompts[3], prompts[4], prompts

                btn_make.click(
                    _make_story,
                    inputs=[product_in, theme_in, appearance_prefix_in],
                    outputs=[story_out, p1_out, p3_out, p4_out, prompts_state],
                )

            # -------------------------
            # Tab 2: Generate (image only)
            # -------------------------
            with gr.TabItem("2) Generate (Image)"):
                handles = generate.build_generate_tab()
                scene_prompt_box = handles["scene_prompt"]

        # ---- Cross-tab wiring (must be after scene_prompt exists) ----
        def _send_prompt(which, prompts_dict):
            if not prompts_dict:
                raise gr.Error("先に Story Planner で Generate Story + Prompts を押してください。")
            key = {"Scene1": 1, "Scene3": 3, "Scene4": 4}[which]
            return prompts_dict[key]

        btn_send.click(
            _send_prompt,
            inputs=[prompt_pick, prompts_state],
            outputs=[scene_prompt_box],
        )

    app.queue().launch(server_name="127.0.0.1", server_port=7861, show_error=True)


if __name__ == "__main__":
    main()
